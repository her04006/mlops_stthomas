from flask import Flask, request, jsonify, render_template
from analyze import get_sentiment, compute_embeddings, classify_email
import json
import logging

app = Flask(__name__, template_folder='templates')

#create logging file to demonstrate add_class and remove_class
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a') # appends to file

@app.route("/")
def home():
    print("Home page")
    return render_template('index.html')


@app.route("/api/v1/sentiment-analysis/", methods=['POST'])
def analysis():
    if request.is_json:
        data = request.get_json()
        sentiment = get_sentiment(data['text'])
        return jsonify({"message": "Data received", "data": data, "sentiment": sentiment}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/valid-embeddings/", methods=['GET'])
def valid_embeddings():
    embeddings = compute_embeddings()
    formatted_embeddings = []
    for text, vector in embeddings:
        formatted_embeddings.append({
            "text": text,
            "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
        })
    embeddings = formatted_embeddings
    return jsonify({"message": "Valid embeddings fetched", "embeddings": embeddings}), 200


@app.route("/api/v1/classify/", methods=['POST'])
def classify():
    if request.is_json:
        data = request.get_json()
        text = data['text']
        logging.info(f"Classifying the following: {text}")
        classifications = classify_email(text)
        return jsonify({"message": "Email classified", "classifications": classifications}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/classify-email/", methods=['GET'])
def classify_with_get():
    text = request.args.get('text')
    classifications = classify_email(text)
    return jsonify({"message": "Email classified", "classifications": classifications}), 200
    
    
@app.route("/api/v1/classes/add/", methods=['POST'])
def add_class():
    "Add functionality to add class"
    if request.is_json:
        data = request.get_json()
        new_class = data.get('class') # user must pass in {"class":"SOME CLASS"}
        logging.info(f"Adding class: {new_class}")
        #print(new_class)
        
        if not new_class:
            return jsonify({"error": "No class provided"}), 400
        
        # Load classes
        try:
            with open('classes.json', 'r') as file:
                class_data = json.load(file)
                logging.info("Loading existing classes...")
                #print(class_data)
        except FileNotFoundError:
            class_data = {"classes": []}
        
        # Add new class if it doesn't exist
        if new_class not in class_data["classes"]: # check if class exists
            class_data["classes"].append(new_class)
            #print(class_data)
            
            with open('classes.json', 'w') as file:
                json.dump(class_data, file)
            logging.info(f"Successfully added new class: {new_class}")
            return jsonify({"message": f"Class '{new_class}' added", "classes": class_data["classes"]}), 200 # on successful add
        else:
            return jsonify({"message": f"Class '{new_class}' already exists", "classes": class_data["classes"]}), 201 # class exist
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400
        
        
# build functionality to remove classes when user wants to remove a class
# code whisperer was used to generate the following by replicating add_class and using .remove instead of .append
@app.route("/api/v1/classes/remove/", methods=['POST'])
def remove_class():
    "Remove functionality to remove class"
    if request.is_json:
        data = request.get_json()
        class_to_remove = data.get('class') # user must pass in {"class":"SOME CLASS"}
        logging.info(f"Removing class: {class_to_remove}")
        #print(class_to_remove)

        if not class_to_remove:
            return jsonify({"error": "No class provided"}), 400

        # Load classes
        try:
            with open('classes.json', 'r') as file:
                class_data = json.load(file)
                #print(class_data)
        except FileNotFoundError:
            class_data = {"classes": []}

        # Remove class if it exists
        if class_to_remove in class_data["classes"]: # check if class exists
            class_data["classes"].remove(class_to_remove)
            #print(class_data)

            with open('classes.json', 'w') as file:
                json.dump(class_data, file)
                
            logging.info(f"Successfully removed class: {class_to_remove}")

            return jsonify({"message": f"Class '{class_to_remove}' removed", "classes": class_data["classes"]}), 200 # on successful removal
        else:
            return jsonify({"message": f"Class '{class_to_remove}' does not exist", "classes": class_data["classes"]}), 404 # class does not exist
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)