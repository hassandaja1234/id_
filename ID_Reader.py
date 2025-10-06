import cv2
import tempfile
from passporteye import read_mrz

def extract_selected_mrz_data(cv2_image):
    """
    Extract selected MRZ fields from a cv2 image using passporteye.

    Args:
        cv2_image (numpy.ndarray): OpenCV image object.

    Returns:
        dict or None: Dictionary with selected MRZ fields if found, else None.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_path = tmp_file.name
        cv2.imwrite(temp_path, cv2_image)

    # Step 2: Read MRZ
    mrz = read_mrz(temp_path)

    # Step 3: Extract selected fields
    if mrz:
        data = mrz.to_dict()

        # Format date of birth and expiration date (YYMMDD → YYYY-MM-DD)
        def format_date(yyMMdd):
            if not yyMMdd or len(yyMMdd) != 6:
                return None
            year_prefix = "20" if int(yyMMdd[:2]) > 30 else "20"
            return f"{year_prefix}{yyMMdd[:2]}-{yyMMdd[2:4]}-{yyMMdd[4:]}"
        def get_national_nunber(data):
            national_number = ''.join(filter(str.isdigit, data))
            return national_number
        return {
            "country": data.get("country"),
            "id_type": data.get("type"),
            "national_number": get_national_nunber(data.get("optional1", "")),
            "date_of_birth": format_date(data.get("date_of_birth")),
            "expiration_date": format_date(data.get("expiration_date")),
            "full_name": f"{data.get('names', '')} {data.get('surname', '')}".strip(),
            "sex": data.get("sex")
        }
    else:
        return None
