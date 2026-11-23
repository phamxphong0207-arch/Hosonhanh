import io
import os
import re
import smtplib
import time
import unicodedata
from email import encoders
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Shinhan Finance - Master 6 Portal", page_icon="🔹", layout="centered")

AI_MAX_WIDTH = 800
COMPANY_MAX_WIDTH = 1200
TIMEOUT_MS = 45000

# Các mục tài liệu (mỗi mục thành 1 file PDF đính kèm mail)
DANH_MUC = ["1. CCCD", "2. Ảnh chụp KH", "3. Lương", "4. Map", "5. VNEID", "6. VSSID", "7. Khác"]
MUC_CONG_TY = "8. Thông tin công ty"
MUC_AI = ["1. CCCD", "5. VNEID"]  # các mục AI đọc để lấy thông tin khách hàng


def get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, "")
    except Exception:
        v = ""
    return v or os.getenv(name, default)


MY_EMAIL = get_secret("MY_EMAIL")
MY_PASSWORD = get_secret("MY_PASSWORD")  # Mật khẩu ứng dụng Gmail (App Password)
RECEIVER_EMAIL = get_secret("RECEIVER_EMAIL")
GEMINI_KEY = get_secret("GEMINI_API_KEY")

st.markdown("""
    <style>
    .stButton>button { background-color: #004686; color: white; width: 100%; border-radius: 8px; font-weight: bold; padding: 10px; }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)


# ----------------------------- Schema AI -----------------------------
class HoSo(BaseModel):
    ho_ten: str = ""
    ngay_sinh: str = ""
    so_cccd: str = ""
    sdt_kh: str = ""
    ngay_cap_cccd: str = ""
    ngay_het_han_cccd: str = ""
    dia_chi_hien_tai: str = ""
    dia_chi_ho_khau: str = ""
    ten_cong_ty: str = ""
    mst_cong_ty: str = ""
    dia_chi_cong_ty: str = ""


PROMPT_BASE = """Bạn là hệ thống OCR hồ sơ vay tại Việt Nam. Trích xuất chính xác, không bịa.
Trường nào không có thì để chuỗi rỗng. Ngày định dạng dd/mm/yyyy.

ẢNH KHÁCH HÀNG (CCCD mặt trước, CCCD mặt sau, VNeID thông tin cá nhân, VNeID thông tin cư trú):
- ho_ten: viết HOA có dấu như trên CCCD.
- ngay_cap_cccd: ngày ở mặt sau CCCD (cạnh Cục trưởng Cục Cảnh sát QLHC về TTXH).
- ngay_het_han_cccd: "Có giá trị đến" ở mặt trước.
- dia_chi_ho_khau: Nơi thường trú trên CCCD (ghép đủ các dòng).
- dia_chi_hien_tai: "Nơi ở hiện tại" trên VNeID thông tin cư trú.
- sdt_kh: số điện thoại trên VNeID thông tin cá nhân.
"""
PROMPT_COMPANY = """
ẢNH CÔNG TY (ảnh ĐẦU TIÊN, trang tra cứu thông tin thuế doanh nghiệp):
- ten_cong_ty: tên doanh nghiệp, viết HOA đúng như ảnh.
- mst_cong_ty: Mã số thuế, chép đúng từng chữ số (có thể có đuôi -001...).
- dia_chi_cong_ty: nội dung dòng "Địa chỉ Thuế", chép nguyên văn.
"""
PROMPT_NO_COMPANY = "\nKhông có ảnh công ty: để trống ten_cong_ty, mst_cong_ty, dia_chi_cong_ty.\n"


# ----------------------------- Xử lý ảnh / PDF -----------------------------
def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", strip_accents(s)).strip("_")
    return s or "KhachHang"


def open_rgb(raw: bytes) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    return img.convert("RGB")


def to_jpeg(raw: bytes, max_side: int, quality: int, by_width: bool = False) -> bytes:
    img = open_rgb(raw)
    if by_width:
        if img.width > max_side:
            img = img.resize((max_side, int(img.height * max_side / img.width)), Image.LANCZOS)
    elif max(img.size) > max_side:
        r = max_side / max(img.size)
        img = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def build_pdf(images: List[bytes]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    a4_w, a4_h = A4
    margin = 20
    for raw in images:
        reader = ImageReader(io.BytesIO(to_jpeg(raw, 1600, 60)))
        w, h = reader.getSize()
        ratio = min((a4_w - 2 * margin) / w, (a4_h - 2 * margin) / h)
        fw, fh = w * ratio, h * ratio
        c.drawImage(reader, (a4_w - fw) / 2, (a4_h - fh) / 2, width=fw, height=fh)
        c.showPage()
    c.save()
    return buf.getvalue()


# ----------------------------- Gemini -----------------------------
@st.cache_resource(show_spinner=False)
def make_client(key: str):
    # Giữ client sống suốt phiên để SDK không đóng kết nối giữa chừng
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))


def extract(key: str, model: str, company_img: Optional[bytes], images: List[bytes]) -> HoSo:
    parts = []
    if company_img:
        parts.append("ẢNH CÔNG TY (tra cứu thuế):")
        parts.append(types.Part.from_bytes(data=company_img, mime_type="image/jpeg"))
    parts.append("ẢNH KHÁCH HÀNG:")
    parts += [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images]
    prompt = PROMPT_BASE + (PROMPT_COMPANY if company_img else PROMPT_NO_COMPANY)
    client = make_client(key)
    resp = client.models.generate_content(
        model=model,
        contents=parts + [prompt],
        config=types.GenerateContentConfig(
            temperature=0.1, response_mime_type="application/json", response_schema=HoSo),
    )
    return resp.parsed or HoSo.model_validate_json(resp.text)


TEMPLATE = """TEAM: {team_name}
KHÁCH HÀNG: {ho_ten}

========================
THÔNG TIN CHI TIẾT HỒ SƠ:
========================
- Ngày sinh: {ngay_sinh}
- Số CCCD: {so_cccd}
- SĐT KH: {sdt_kh}
- Ngày cấp CCCD: {ngay_cap_cccd}
- Ngày hết hạn CCCD: {ngay_het_han_cccd}
- Địa chỉ hiện tại: {dia_chi_hien_tai}
- Địa chỉ hộ khẩu: {dia_chi_ho_khau}
- Tên công ty: {ten_cong_ty}
- Mã số thuế công ty: {mst_cong_ty}
- Địa chỉ công ty: {dia_chi_cong_ty}
- SĐT Tham chiếu (nếu có): {sdt_tham_chieu}
- Thu nhập (Lương): {thu_nhap_luong}
- Nhu cầu vay: {nhu_cau_vay}
- Tên NV: {ten_nv}"""


# ----------------------------- Gửi mail -----------------------------
def send_mail(subject: str, body: str, attachments: list):
    """attachments: [(tên_file, bytes, subtype)]"""
    if not (MY_EMAIL and MY_PASSWORD and RECEIVER_EMAIL):
        raise RuntimeError("Chưa cấu hình MY_EMAIL / MY_PASSWORD / RECEIVER_EMAIL trong Secrets.")
    msg = MIMEMultipart()
    msg["From"] = MY_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = Header(subject, "utf-8").encode()
    msg.attach(MIMEText(body, "plain", "utf-8"))
    for name, data, subtype in attachments:
        part = MIMEBase("application", subtype)
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(part)
    server = smtplib.SMTP("smtp.gmail.com", 587, timeout=60)
    try:
        server.starttls()
        server.login(MY_EMAIL, MY_PASSWORD)
        server.send_message(msg)
    finally:
        server.quit()


# ----------------------------- Ảnh minh họa (tuỳ chọn) -----------------------------
GUIDE = [
    ("cong_ty", "🏢", "Ảnh công ty"), ("cccd_truoc", "🪪", "CCCD mặt trước"),
    ("cccd_sau", "🪪", "CCCD mặt sau"), ("vneid_cn", "📱", "VNeID cá nhân"),
    ("vneid_cutru", "📱", "VNeID cư trú"),
]


def find_guide_image(key: str) -> Optional[str]:
    base = os.path.dirname(os.path.abspath(__file__))
    for folder in (os.path.join(base, "guide"), base):
        for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
            p = os.path.join(folder, f"{key}.{ext}")
            if os.path.exists(p):
                return p
    return None


def render_guide():
    with st.expander("📋 Xem ảnh mẫu: chụp những màn hình nào?"):
        for i in range(0, len(GUIDE), 3):
            cols = st.columns(3)
            for col, (key, icon, title) in zip(cols, GUIDE[i:i + 3]):
                with col:
                    p = find_guide_image(key)
                    if p:
                        st.image(p, use_container_width=True)
                    else:
                        st.markdown(f"<div style='text-align:center;font-size:34px'>{icon}</div>",
                                    unsafe_allow_html=True)
                    st.caption(title)


# ----------------------------- Giao diện -----------------------------
with st.sidebar:
    st.header("Cấu hình")
    api_key = GEMINI_KEY or st.text_input("Gemini API Key", type="password")
    model = st.selectbox("Model AI", ["gemini-3.5-flash-lite", "gemini-3.8-flash"], index=0)
    team = st.text_input("TEAM", value="MASTER 6")
    ten_nv = st.text_input("Tên Nhân Viên (NV)", value="Nguyễn Văn Cường")
    auto_send = st.checkbox("Tự động gửi mail ngay sau khi trích xuất", value=True,
                            help="Bỏ chọn để xem lại và sửa nội dung trước khi gửi.")

if os.path.exists("logo.png"):
    st.image("logo.png", width=200)
else:
    st.title("🔹 SHINHAN FINANCE")
st.subheader("TEAM MASTER 6 - CỔNG NHẬP HỒ SƠ ONLINE")

render_guide()

st.write("### 1. Thông tin bổ sung")
ten_kh_nhap = st.text_input("Họ tên khách hàng (để trống nếu muốn AI tự đọc từ CCCD):")
c1, c2 = st.columns(2)
sdt_tc = c1.text_input("SĐT Tham chiếu")
nhu_cau = c2.text_input("Nhu cầu vay")
luong = st.text_input("Thu nhập/Lương (ghi đè nếu cần)")

st.write("### 2. Ảnh hồ sơ")
st.info("💡 Trên điện thoại có thể chọn ảnh từ album hoặc chụp trực tiếp bằng máy ảnh.")
file_storage = {}
cty = st.file_uploader("🏢 Thông tin công ty (trang tra cứu thuế: tên + MST + Địa chỉ Thuế)",
                       type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="cty")
file_storage[MUC_CONG_TY] = [cty] if cty else []
for muc in DANH_MUC:
    file_storage[muc] = st.file_uploader(
        f"Tải lên hoặc chụp ảnh cho: {muc}", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True, key=muc) or []

run = st.button("🚀 TRÍCH XUẤT & GỬI MAIL")


def do_send(text: str, ho_ten: str):
    """Tạo PDF từng mục + file txt rồi gửi mail."""
    name = safe_name(ho_ten)
    attachments = []
    with st.spinner("Đang nén ảnh, tạo PDF và gửi mail..."):
        for muc, files in file_storage.items():
            if files:
                pdf = build_pdf([f.getvalue() for f in files])
                attachments.append((f"{name}_{safe_name(muc.split('.', 1)[-1])}.pdf", pdf, "pdf"))
        attachments.append((f"{name}_ThongTin.txt", text.encode("utf-8"), "plain"))
        try:
            send_mail(f"Hồ sơ {ho_ten}", text, attachments)
            st.success(f"🎉 Đã gửi hồ sơ khách hàng {ho_ten} kèm {len(attachments) - 1} file PDF.")
        except Exception as e:
            st.error(f"❌ Gửi mail thất bại: {e}")


send_now = False
if run:
    ai_files = [f for m in MUC_AI for f in file_storage[m]][:6]
    if not ai_files:
        st.warning("⚠️ Vui lòng tải ảnh CCCD (mục 1) và VNEID (mục 5) để AI đọc thông tin.")
    elif not api_key:
        st.error("Chưa có Gemini API Key (đặt GEMINI_API_KEY trong Secrets hoặc nhập ở thanh bên).")
    else:
        with st.status("Đang xử lý...", expanded=True) as status:
            try:
                t0 = time.time()
                imgs = [to_jpeg(f.getvalue(), AI_MAX_WIDTH, 85, by_width=True) for f in ai_files]
                comp = to_jpeg(cty.getvalue(), COMPANY_MAX_WIDTH, 85, by_width=True) if cty else None
                st.write(f"Bước 1: Nén ảnh - {time.time() - t0:.2f}s")
                t1 = time.time()
                hs = extract(api_key, model, comp, imgs)
                st.write(f"Bước 2: Gemini ({model}) - {time.time() - t1:.2f}s")
                if not comp:
                    st.warning("Chưa có ảnh công ty: tên công ty, MST, địa chỉ công ty sẽ để trống.")
                status.update(label=f"Trích xuất xong trong {time.time() - t0:.2f}s", state="complete")
            except Exception as e:
                status.update(label="Lỗi", state="error")
                st.exception(e)
                st.stop()
        ho_ten = ten_kh_nhap.strip() or hs.ho_ten or "KhachHang"
        text = TEMPLATE.format(
            team_name=team, ho_ten=ho_ten, ngay_sinh=hs.ngay_sinh, so_cccd=hs.so_cccd,
            sdt_kh=hs.sdt_kh, ngay_cap_cccd=hs.ngay_cap_cccd,
            ngay_het_han_cccd=hs.ngay_het_han_cccd, dia_chi_hien_tai=hs.dia_chi_hien_tai,
            dia_chi_ho_khau=hs.dia_chi_ho_khau, ten_cong_ty=hs.ten_cong_ty,
            mst_cong_ty=hs.mst_cong_ty, dia_chi_cong_ty=hs.dia_chi_cong_ty,
            sdt_tham_chieu=sdt_tc, thu_nhap_luong=luong, nhu_cau_vay=nhu_cau, ten_nv=ten_nv)
        st.session_state["draft"] = {"text": text, "ho_ten": ho_ten, "id": time.time()}
        send_now = auto_send

draft = st.session_state.get("draft")
if draft:
    st.write("### 3. Nội dung hồ sơ")
    edited = st.text_area("Nội dung (có thể sửa trước khi gửi):", value=draft["text"],
                          height=420, key=f"txt_{draft['id']}")
    if send_now:
        do_send(draft["text"], draft["ho_ten"])
    elif st.button("📧 Gửi mail với nội dung trên"):
        do_send(edited, draft["ho_ten"])
