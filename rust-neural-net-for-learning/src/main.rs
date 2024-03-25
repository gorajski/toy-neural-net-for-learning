const LEARNING_RATE: f64 = 0.1;

fn main() {
    // get hidden and output layer from setup function
    let (mut hidden_layer, mut output_layer) = setup();

    // XOR
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // Train the model using the example data
    train_network(&inputs, &targets, &mut hidden_layer, &mut output_layer);

    println!("Training Result: {:?}, {:?}", hidden_layer, output_layer);

    // Flex it!
    for (i, input) in inputs.iter().enumerate() {
        let prediction = predict(&input, &hidden_layer, &output_layer);
        println!("prediction for {:?}: {}, target: {:?}", input, prediction, &targets[i]);
    }
}

fn train_network(
    inputs: &Vec<Vec<f64>>,
    targets: &Vec<Vec<f64>>,
    hidden_layer: &mut Layer,
    output_layer: &mut Layer,
) {
    let iterations = 1000000;
    for _ in 0..iterations {
        for i in 0..inputs.len() {
            backprop(
                &inputs[i],
                &targets[i],
                hidden_layer,
                output_layer,
                LEARNING_RATE,
            );
            // println!(
            //     "Backprop result {}: {:?}, {:?}",
            //     i, hidden_layer, output_layer
            // );
        }
    }
}

fn setup() -> (Layer, Layer) {
    let hidden_layer = Layer {
        neurons: vec![
            Neuron {
                weights: vec![0.5, 0.5],
                bias: 0.0,
            },
            Neuron {
                weights: vec![0.5, 0.5],
                bias: 0.0,
            },
        ],
    };

    let output_layer = Layer {
        neurons: vec![Neuron {
            weights: vec![0.5, 0.5],
            bias: 0.0,
        }],
    };

    (hidden_layer, output_layer)
}

#[derive(Debug)]
struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn eval(&self, inputs: &Vec<f64>) -> f64 {
        let dot_product = dot_product(&self.weights, &inputs);
        let weighted_sum = dot_product + self.bias;
        sigmoid(weighted_sum)
    }
}

#[derive(Debug)]
struct Layer {
    neurons: Vec<Neuron>,
}

fn predict(input: &Vec<f64>, hidden_layer: &Layer, output_layer: &Layer) -> f64 {
    let hidden_outputs: Vec<f64> = hidden_layer
        .neurons
        .iter()
        .map(|neuron| neuron.eval(input))
        .collect();
    output_layer.neurons[0].eval(&hidden_outputs)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn signmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}

fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter()
        .enumerate()
        .fold(0.0, |acc, (i, component)| acc + (component * b[i]))
}

fn backprop(
    inputs: &Vec<f64>,
    targets: &Vec<f64>,
    hidden_layer: &mut Layer,
    output_layer: &mut Layer,
    learning_rate: f64,
) {
    // Calculate the outputs of the hidden layer
    let hidden_outputs: Vec<f64> = hidden_layer
        .neurons
        .iter()
        .map(|neuron| neuron.eval(inputs))
        .collect();

    // Calculate the outputs of the output layer
    let output_dot_product = dot_product(&output_layer.neurons[0].weights, &hidden_outputs);
    let output = output_dot_product + output_layer.neurons[0].bias;
    let output_error = (targets[0] - output) * signmoid_derivative(output);

    // Calculate the errors of the hidden layer
    let hidden_errors: Vec<f64> = hidden_layer
        .neurons
        .iter()
        .map(|neuron| {
            let output_weighted_sum = output_layer.neurons[0]
                .weights
                .iter()
                .fold(0.0, |acc, weight| acc + (weight * output_error));

            let hidden_layer_weighted_sum = neuron
                .weights
                .iter()
                .enumerate()
                .fold(0.0, |acc, (i, weight)| acc + (weight * inputs[i]))
                + neuron.bias;

            output_weighted_sum * signmoid_derivative(sigmoid(hidden_layer_weighted_sum))
        })
        .collect();

    // Update the weights and biases of the output layer
    for neuron in &mut output_layer.neurons {
        neuron.weights = neuron
            .weights
            .iter()
            .enumerate()
            .map(|(j, weight)| weight + (learning_rate * output_error * hidden_outputs[j]))
            .collect();

        neuron.bias += learning_rate * output_error;
    }

    // Update the weights and biases of the hidden layer
    for (i, neuron) in hidden_layer.neurons.iter_mut().enumerate() {
        neuron.weights = neuron
            .weights
            .iter()
            .enumerate()
            .map(|(j, weight)| weight + (learning_rate * hidden_errors[i] * inputs[j]))
            .collect();

        neuron.bias += learning_rate * hidden_errors[i];
    }
}

#[test]
fn run_a_test() {
    assert_eq!(1, 1)
}
