# In this file, we define run_model
# It runs every time the server is called

def run_model(model, prompt):
    
    result = model.generate(prompt)
    
    return result