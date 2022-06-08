from sanic import Sanic, response
from sanic.response import json as json_response
from warmup import load_model
from run import run_model
from transformers import GPT2Tokenizer

# do the warmup step globally, to have a reuseable model instance
model = load_model()
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")

app = Sanic("my_app")


@app.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    return response.json({"state": "healthy"})

@app.route('/', methods=["POST"]) # Do not edit - POST requests to "/" are a required interface
def inference(request):
    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return response.json({'message': "No prompt provided"})

    input_tokens = tokenizer.encode(input, return_tensors="pt").to("cuda:0")
    output = run_model(model, input_tokens)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    response = {"output": output_text}
    return json_response(response) # Do not edit - returning a dictionary as JSON is a required interface


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000", workers=1)