from ultralytics import YOLO, checks, hub
checks()

hub.login('6bdf707bab845ab6ce7ff34ba396a9b90d3f7cd731')

model = YOLO('https://hub.ultralytics.com/models/cG1kEK2ejSxvDHhUxrwu')
results = model.train()