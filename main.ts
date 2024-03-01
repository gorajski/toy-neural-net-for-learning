import Neuron from "./neuron";

const activationFunction = (n: number) => Math.exp(n) / (1 + Math.exp(n));

const neuron = new Neuron(activationFunction, [
  Math.random(),
  Math.random(),
  Math.random(),
], -0.1);

const trainingData = [
  {
    inputs: [-1, -1, -1],
    desiredOutput: -1,
  },
  {
    inputs: [1, -1, -1],
    desiredOutput: -1,
  },
  {
    inputs: [-1, 1, -1],
    desiredOutput: -1,
  },
  {
    inputs: [-1, -1, 1],
    desiredOutput: -1,
  },
  {
    inputs: [-1, 1, 1],
    desiredOutput: -1,
  },
  {
    inputs: [1, -1, 1],
    desiredOutput: 1,
  },
  {
    inputs: [1, 1, -1],
    desiredOutput: 1,
  },
  {
    inputs: [1, 1, 1],
    desiredOutput: 1,
  },
];

for (let i = 0; i < 1000000; i++) {
  trainingData.forEach((data) => {
    neuron.update(data.inputs, data.desiredOutput);
  });

  if (i % 1000 === 0) {
    console.log("iteration:", i);
    console.log(neuron);
  }
}

console.log("### FINAL:");
trainingData.forEach((data) => {
  console.log(data.inputs, neuron.eval(data.inputs));
});
