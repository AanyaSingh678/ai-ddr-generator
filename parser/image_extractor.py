"""
parser/image_extractor.py

Extracts unique images from a PDF file.
Deduplicates using MD5 hashing to avoid saving the same image twice.
Each extracted image is returned with its page number and nearby headings
so downstream modules can map images to the correct DDR section.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path


import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """
    Represents a single image extracted from a PDF page.

    Attributes:
        file_path:    Absolute path to the saved image file.
        page_number:  1-indexed page where the image was found.
        image_index:  Position of the image on that page (0-indexed).
        image_hash:   MD5 hash of the raw image bytes (used for deduplication).
        width:        Image width in pixels.
        height:       Image height in pixels.
        extension:    File extension (e.g., 'png', 'jpeg').
        nearby_text:  First 200 characters of text on the same page.
                      Used to map the image to the correct DDR section.
    """

    file_path: Path
    page_number: int
    image_index: int
    image_hash: str
    width: int
    height: int
    extension: str
    nearby_text: str = ""

    def __repr__(self) -> str:
        return (
            f"ExtractedImage(page={self.page_number}, "
            f"index={self.image_index}, "
            f"size={self.width}x{self.height}, "
            f"file='{self.file_path.name}')"
        )


@dataclass
class ExtractionResult:
    """
    Summary of an image extraction run.

    Attributes:
        images:           List of successfully extracted unique images.
        total_found:      Total image references found in the PDF (including duplicates).
        duplicates_skipped: Count of images skipped due to hash match.
        source_pdf:       Path of the source PDF that was processed.
    """

    images: list[ExtractedImage] = field(default_factory=list)
    total_found: int = 0
    duplicates_skipped: int = 0
    source_pdf: Path | None = None

    @property
    def unique_count(self) -> int:
        return len(self.images)

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(source='{self.source_pdf.name if self.source_pdf else '?'}', "
            f"unique={self.unique_count}, "
            f"duplicates_skipped={self.duplicates_skipped})"
        )


class ImageExtractor:
    """
    Extracts all unique images from a PDF file and saves them to disk.

    Deduplication Strategy:
        Each image's raw bytes are hashed with MD5.
        If the same hash has been seen before within this extraction run,
        the image is skipped. This handles PDFs like UrbanRoof reports
        that embed the same logo or background graphic on every page.

    Page Metadata:
        Each image is tagged with its page number and nearby page text.
        This allows the DDR report generator to place images under the
        correct area-wise observation section.

    Usage:
        extractor = ImageExtractor(
            pdf_path="data/inspection_report.pdf",
            output_folder="images/inspection"
        )
        result = extractor.extract()
        print(result)
        for img in result.images:
            print(img)
    """

    # Images smaller than this in either dimension are likely icons or
    # decorative elements (e.g. logos, watermarks). Skip them.
    MIN_DIMENSION_PX = 50

    def __init__(self, pdf_path: str, output_folder: str) -> None:
        self.pdf_path = Path(pdf_path)
        self.output_folder = Path(output_folder)
        self._validate_inputs()
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _validate_inputs(self) -> None:
        """
        Validates that the PDF path exists and is a .pdf file.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError: If the file is not a PDF.
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: '{self.pdf_path}'. "
                f"Ensure the file is placed inside the /data folder."
            )
        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a .pdf file, got: '{self.pdf_path.suffix}'"
            )

    @staticmethod
    def _compute_hash(image_bytes: bytes) -> str:
        """
        Computes an MD5 hash of image bytes for deduplication.

        MD5 is used here for speed, not security. It is sufficient
        for detecting exact duplicate images within a single document.

        Args:
            image_bytes: Raw bytes of the image.

        Returns:
            Hexadecimal MD5 digest string.
        """
        return hashlib.md5(image_bytes).hexdigest()

    @staticmethod
    def _is_too_small(width: int, height: int, threshold: int) -> bool:
        """
        Returns True if the image is smaller than the threshold in either dimension.
        Used to filter out logos, icons, and decorative elements.
        """
        return width < threshold or height < threshold

    @staticmethod
    def _get_nearby_text(page: fitz.Page) -> str:
        """
        Extracts the first 200 characters of text from a page.

        This text is stored with each image so the report generator
        can determine which DDR section the image belongs to.

        Args:
            page: A PyMuPDF page object.

        Returns:
            Trimmed string of nearby text (up to 200 characters).
        """
        text = page.get_text().strip()
        return text[:200] if text else ""

    def _save_image(
        self,
        image_bytes: bytes,
        extension: str,
        page_number: int,
        image_index: int,
    ) -> Path:
        """
        Saves image bytes to disk and returns the file path.

        Filename format: page_{page_number}_img_{image_index}.{ext}
        Page number is 1-indexed to match human-readable page references.

        Args:
            image_bytes:  Raw bytes of the image to save.
            extension:    File extension without dot (e.g., 'png').
            page_number:  1-indexed page number.
            image_index:  0-indexed image position on the page.

        Returns:
            Path to the saved image file.
        """
        filename = f"page_{page_number}_img_{image_index}.{extension}"
        image_path = self.output_folder / filename

        with open(image_path, "wb") as image_file:
            image_file.write(image_bytes)

        return image_path

    def extract(self) -> ExtractionResult:
        """
        Extracts all unique images from the PDF.

        Iterates over every page, extracts all embedded images,
        deduplicates by hash, filters out tiny images, and saves
        each unique image to the output folder.

        Returns:
            ExtractionResult containing all extracted images and summary stats.

        Raises:
            RuntimeError: If the PDF is encrypted or cannot be opened.
        """
        logger.info("Starting image extraction: '%s'", self.pdf_path.name)

        result = ExtractionResult(source_pdf=self.pdf_path)
        seen_hashes: set[str] = set()

        try:
            with fitz.open(self.pdf_path) as document:

                if document.is_encrypted:
                    raise RuntimeError(
                        f"PDF is encrypted and cannot be processed: '{self.pdf_path.name}'"
                    )

                for page_index, page in enumerate(document):
                    page_number = page_index + 1  # convert to 1-indexed
                    nearby_text = self._get_nearby_text(page)
                    image_list = page.get_images(full=True)

                    for img_index, img_ref in enumerate(image_list):
                        result.total_found += 1

                        xref = img_ref[0]

                        try:
                            base_image = document.extract_image(xref)
                        except Exception as e:
                            logger.warning(
                                "Failed to extract image xref=%d on page %d: %s",
                                xref, page_number, e
                            )
                            continue

                        image_bytes: bytes = base_image["image"]
                        extension: str = base_image["ext"]
                        width: int = base_image.get("width", 0)
                        height: int = base_image.get("height", 0)

                        # Filter out tiny decorative images (logos, watermarks)
                        if self._is_too_small(width, height, self.MIN_DIMENSION_PX):
                            logger.debug(
                                "Skipping small image (%dx%d) on page %d.",
                                width, height, page_number
                            )
                            continue

                        # Deduplicate by hash
                        image_hash = self._compute_hash(image_bytes)
                        if image_hash in seen_hashes:
                            result.duplicates_skipped += 1
                            logger.debug(
                                "Duplicate image skipped on page %d (hash: %s).",
                                page_number, image_hash
                            )
                            continue

                        seen_hashes.add(image_hash)

                        # Save to disk
                        saved_path = self._save_image(
                            image_bytes, extension, page_number, img_index
                        )

                        extracted_image = ExtractedImage(
                            file_path=saved_path,
                            page_number=page_number,
                            image_index=img_index,
                            image_hash=image_hash,
                            width=width,
                            height=height,
                            extension=extension,
                            nearby_text=nearby_text,
                        )

                        result.images.append(extracted_image)

                        logger.debug("Saved: %s", extracted_image)

        except fitz.FileDataError as e:
            raise RuntimeError(
                f"Could not open PDF '{self.pdf_path.name}'. "
                f"The file may be corrupted. Details: {e}"
            ) from e

        logger.info(
            "Extraction complete: %d unique images saved, "
            "%d duplicates skipped, %d total found.",
            result.unique_count,
            result.duplicates_skipped,
            result.total_found,
        )

        return result

    def __repr__(self) -> str:
        return (
            f"ImageExtractor("
            f"pdf='{self.pdf_path.name}', "
            f"output='{self.output_folder}')"
        )