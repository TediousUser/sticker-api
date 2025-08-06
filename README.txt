This version gives output as a response json where the image is converted into base64 form.

Response JSON:

{
  "message": "OK, Sticker present",
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}


How to run:
To run the app:
1. uvicorn api:app --reload --host 0.0.0.0 --port 8000
To get response
2. curl -X POST "http://localhost:8000/infer" \
     -F "file=@test.jpg" \
     -o response.json


Try out the exampleUI:
After running the app, simply double-click and open the exampleUI.html in browser


