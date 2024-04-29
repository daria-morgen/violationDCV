from unittest import TestCase

from app.handlers.violatation_dcv import ViolatationDCV


class TestViolatationDCV(TestCase):

    def test_detect_fail_no_humans(self):

        app = ViolatationDCV()

        with self.assertRaises(Exception):
            app.detect(1,'https://cdn.iz.ru/sites/default/files/news-2024-01/1_4_4.jpg')


