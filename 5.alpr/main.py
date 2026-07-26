from pprint import pprint

from dataset.parsers.ir_lpr_xml_parser import IRLPRXMLParser

parser = IRLPRXMLParser()

annotation = parser.parse(
    "data/car_images/train/day_00001.xml"
)

print(annotation.image)

pprint(annotation.objects)