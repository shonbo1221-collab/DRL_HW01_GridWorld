from flask import Flask, render_template, request, jsonify
from rl_env import evaluate_policy, value_iteration

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    n = data.get('n', 5)
    end_idx = data.get('end', -1)
    obstacles = data.get('obstacles', [])
    
    if end_idx == -1: return jsonify({'error': 'No end cell selected.'}), 400
    policy, values = evaluate_policy(n, end_idx, obstacles)
    return jsonify({'policy': policy, 'values': values})

@app.route('/api/value_iteration', methods=['POST'])
def val_iter():
    data = request.json
    n = data.get('n', 5)
    end_idx = data.get('end', -1)
    obstacles = data.get('obstacles', [])
    
    if end_idx == -1: return jsonify({'error': 'No end cell selected.'}), 400
    policy, values = value_iteration(n, end_idx, obstacles)
    return jsonify({'policy': policy, 'values': values})

if __name__ == '__main__':
    app.run(debug=True)
