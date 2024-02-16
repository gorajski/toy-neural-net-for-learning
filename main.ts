import Neuron from './neuron'
const activationFunction = (n: number) => Math.exp(n) / (1 + Math.exp(n))
const neuron = new Neuron(activationFunction, [Math.random(), Math.random(), Math.random()], -5)
const trainingData = [
    {
        inputs: [-1, -1, -1],
        desiredOutput: -1
    },
    {
        inputs: [1, -1, -1],
        desiredOutput: 1
    },
    {
        inputs: [-1, 1, -1],
        desiredOutput: -1
    },
    {
        inputs: [-1, -1, 1],
        desiredOutput: -1
    },
    {
        inputs: [-1, 1, 1],
        desiredOutput: -1
    },
    {
        inputs: [1, -1, 1],
        desiredOutput: -1
    },
    {
        inputs: [1, 1, -1],
        desiredOutput: 1
    },
    {
        inputs: [1, 1, 1],
        desiredOutput: -1
    },
]
for (let i = 0; i < 100000; i++) {
    trainingData.forEach(data => {
        const gradientArray = neuron.backprop(data.inputs, data.desiredOutput)
        // console.log("gradientArray:", gradientArray)
        neuron.updateWeights(gradientArray)
    })

    if (i % 1000 === 0) {
        console.log("iteration:", i)
        console.log(neuron)
    }
}

console.log("### FINAL:")
trainingData.forEach(data => {
    console.log(data.inputs, neuron.eval(data.inputs))
})
