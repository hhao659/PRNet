from ultralytics import YOLO, checks, hub
checks()

hub.login('85503346f98c564248bba2c316446aae7a7242acd0')

model = YOLO('https://hub.ultralytics.com/models/9lDeUFJ9meXmVYXBynLy')
results = model.train()