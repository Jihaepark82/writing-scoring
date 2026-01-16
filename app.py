import io
from typing import Tuple

import streamlit as st
from PIL import Image

from google import genai
from google.genai import types


APP_TITLE = "사진 OCR + 글 평가 (Gemini 2.5 Flash)"
MODEL_NAME = "gemini-2.5-flash"


def get_client() -> genai.Client:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            'GEMINI_API_KEY가 설정되어 있지 않습니다. '
            'Streamlit secrets에 st.secrets["GEMINI_API_KEY"]로 설정하세요.'
        )
    return genai.Client(api_key=api_key)


def pil_to_bytes_and_mime(img: Image.Image) -> Tuple[bytes, str]:
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), "image/jpeg"


def ocr_with_gemini(client: genai.Client, image_bytes: bytes, mime_type: str) -> str:
    prompt = (
        "당신은 고정밀 OCR 엔진입니다.\n"
        "이미지 안의 글자를 가능한 한 정확히 추출하세요.\n"
        "- 보이는 텍스트만 추출 (추측/보완 금지)\n"
        "- 줄바꿈/문단 구조 최대한 유지\n"
        "- 결과는 '추출 텍스트'만 출력 (설명/머리말/코드블록 금지)\n"
    )

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[
                    # ✅ FIX: 키워드 인자로 전달
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=4096,
        ),
    )
    return (resp.text or "").strip()


def evaluate_writing(client: genai.Client, text: str) -> str:
    prompt = f"""
아래 글을 한국어로 평가하고 개선 피드백을 제공하세요.

[평가 기준]
1) 맞춤법/문법: 오류 지적 + 왜 문제인지 + 고친 예시
2) 표현력: 어색한 표현/중복/문장 호흡 개선
3) 내용 풍부성: 구체성, 근거, 예시, 구조(서론-본론-결론) 관점 제안
4) 총평: 장점/우선순위 3가지
5) 개선본: 원문 의미를 크게 바꾸지 않는 선에서 8~15문장 정도로 더 읽기 좋게 다듬은 버전 제시
   (원문이 아주 짧다면 길이를 원문에 맞춰 무리하게 늘리지 말 것)

[출력 형식]
- Markdown
- 섹션 제목 사용: ## 맞춤법/문법, ## 표현력, ## 내용 풍부성, ## 총평(우선순위 3가지), ## 개선본

[원문]
{text}
""".strip()

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Content(
                role="user",
                parts=[
                    # ✅ FIX: 키워드 인자로 전달
                    types.Part.from_text(text=prompt)
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=4096,
        ),
    )
    return (resp.text or "").strip()


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("모델: gemini-2.5-flash (멀티모달 OCR + 글 평가)")

with st.sidebar:
    st.subheader("설정")
    run_ocr = st.checkbox("1) OCR 실행", value=True)
    run_eval = st.checkbox("2) 글 평가 실행", value=True)
    show_raw_ocr = st.checkbox("추출 텍스트(원문) 표시", value=True)
    st.divider()
    st.write("API Key는 `st.secrets['GEMINI_API_KEY']`로만 읽습니다.")

tab_cam, tab_up = st.tabs(["카메라로 찍기", "이미지 업로드"])

image: Image.Image | None = None

with tab_cam:
    cam = st.camera_input("사진을 찍어 업로드하세요", key="camera")
    if cam is not None:
        image = Image.open(cam)

with tab_up:
    up = st.file_uploader("이미지 파일을 업로드하세요 (png/jpg/jpeg)", type=["png", "jpg", "jpeg"], key="upload")
    if up is not None:
        image = Image.open(up)

if image is None:
    st.info("카메라로 촬영하거나 이미지를 업로드하면 OCR 및 글 평가를 진행합니다.")
    st.stop()

col1, col2 = st.columns([1, 1], vertical_alignment="top")
with col1:
    st.subheader("입력 이미지")
    st.image(image, use_container_width=True)

try:
    client = get_client()
except Exception as e:
    st.error(str(e))
    st.stop()

image_bytes, mime_type = pil_to_bytes_and_mime(image)

extracted_text = ""
if run_ocr:
    with st.spinner("OCR(텍스트 추출) 중..."):
        extracted_text = ocr_with_gemini(client, image_bytes, mime_type)

with col2:
    st.subheader("결과")

    if run_ocr:
        if not extracted_text.strip():
            st.warning("추출된 텍스트가 비어 있습니다. 더 선명한 사진/각도/조명을 시도해 보세요.")
        else:
            if show_raw_ocr:
                st.markdown("#### 추출 텍스트")
                st.text_area("OCR 결과", extracted_text, height=220)

    if run_eval:
        base_text = extracted_text.strip() if extracted_text.strip() else ""
        if not base_text:
            st.warning("평가할 텍스트가 없습니다. OCR이 먼저 성공해야 합니다.")
        else:
            with st.spinner("글 평가/피드백 생성 중..."):
                feedback_md = evaluate_writing(client, base_text)
            st.markdown("#### 평가 및 피드백")
            st.markdown(feedback_md)

st.divider()
st.caption("주의: OCR 특성상 원문과 차이가 날 수 있으니, 중요 문서는 반드시 검수하세요.")
