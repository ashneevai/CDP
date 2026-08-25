from __future__ import annotations

import time


class WidthTriggeredOCR:
    def extract(self, image):
        if image.width == 1:
            time.sleep(5)
        return []

    def extract_region(self, image, *_bbox):
        return self.extract(image)
