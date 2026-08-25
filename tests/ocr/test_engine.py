"""Unit tests for Gemini OCR Engine."""

from unittest.mock import MagicMock, patch
import pytest
from ocr.engine import SYSTEM_INSTRUCTION, GeminiOCREngine
from ocr.schemas import LoadedImageData


def test_gemini_engine_init_missing_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
            GeminiOCREngine(api_key=None)


def test_gemini_engine_process_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"공사명": "역삼동 빌딩 신축", "공종": "전기설비", "위치": "지하 2층 EPS실", "내용": "트레이 설치 (400W x 100H, Φ16 전선관)", "일자": "2026.08.25"}'
    mock_client.models.generate_content.return_value = mock_response

    with patch("ocr.engine.genai.Client", return_value=mock_client):
        engine = GeminiOCREngine(api_key="dummy_key_123", model="gemini-3.5-flash-lite")
        image_data = LoadedImageData(
            image_bytes=b"dummy_bytes",
            mime_type="image/jpeg",
            source_id="board_1.jpg",
            file_name="board_1.jpg",
        )

        result = engine.process(image_data)

        assert result.success is True
        assert result.data is not None
        assert result.data.공사명 == "역삼동 빌딩 신축"
        assert result.data.공종 == "전기설비"
        assert "Φ16" in result.data.내용
        assert result.data.일자 == "2026.08.25"

        # Verify generate_content call arguments
        call_args = mock_client.models.generate_content.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["model"] == "gemini-3.5-flash-lite"
        
        # Verify contents has ONLY image part (no prompt text string)
        contents = kwargs["contents"]
        assert len(contents) == 1
        assert not any(isinstance(c, str) for c in contents)


def test_gemini_engine_null_field_handling():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"공사명": "현장A", "공종": null, "위치": null, "내용": "벽체 거푸집 조립", "일자": null}'
    mock_client.models.generate_content.return_value = mock_response

    with patch("ocr.engine.genai.Client", return_value=mock_client):
        engine = GeminiOCREngine(api_key="dummy_key_123")
        image_data = LoadedImageData(
            image_bytes=b"dummy_bytes",
            mime_type="image/jpeg",
            source_id="board_2.jpg",
            file_name="board_2.jpg",
        )

        result = engine.process(image_data)
        assert result.success is True
        assert result.data.공종 is None
        assert result.data.위치 is None
        assert result.data.일자 is None
        assert result.data.공사명 == "현장A"
        assert result.data.내용 == "벽체 거푸집 조립"


def test_gemini_engine_api_key_masking_in_error():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API error with key secret_key_abc_999 occurred")

    with patch("ocr.engine.genai.Client", return_value=mock_client):
        engine = GeminiOCREngine(api_key="secret_key_abc_999")
        image_data = LoadedImageData(
            image_bytes=b"dummy",
            mime_type="image/jpeg",
            source_id="err.jpg",
            file_name="err.jpg",
        )

        result = engine.process(image_data)
        assert result.success is False
        assert "secret_key_abc_999" not in result.error_message
        assert "***API_KEY_MASKED***" in result.error_message
