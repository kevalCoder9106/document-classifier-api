import base64
from flask import Flask, request
import joblib
import os
from base64 import b64decode
import io
import PyPDF2 
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['post'])
def predict():
    data = request.files.to_dict(flat=False)
    file_data = []

    for id in data:
        pdf_bytes = io.BytesIO(data[id][0].read())
        pdf_data = PyPDF2.PdfFileReader(pdf_bytes)
        text = ''
        print(f"Got the text of {data[id]}")

        i = 0
        while True:
            try:
                pageObj = pdf_data.getPage(i) 
                text += pageObj.extractText()
                i += 1
            except:
                break
        file_data.append(text)

    output = []
    for i in range(len(file_data)):
        d = [file_data[i]]
        p_d = pre_process.transform(d).toarray()
        output.append(str(clf.predict(p_d)[0]))

    return json.dumps(output)

if __name__ == '__main__':
    classes = ['Art & Science',
    'Finance',
    'Goverment & Politics',
    'Health',
    'Science & Technology']

#     _port = int(os.environ.get('PORT'))
    pre_process = joblib.load('pre_process.pkl')
    clf = joblib.load('model.pkl')
    app.run(host='0.0.0.0',port=3000)
