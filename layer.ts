import Neuron from "./neuron"
import { ActFunc } from './types'

export default class Layer {
    neurons: Neuron[]
    inputBuffer: number[]
    inputBufferSize: number

    constructor(activationFunction: ActFunc, neuronCount: number, inputBufferSize: number) {
        this.neurons = this.initializeNeurons(
            activationFunction,
            neuronCount,
            inputBufferSize
        )
        this.inputBuffer = []
        this.inputBufferSize = inputBufferSize
    }

    applyInputs(inputs: number[]): void {
        if (inputs.length !== this.inputBufferSize) {
            throw new Error("Layer received incorrect number of inputs")
        }
        this.inputBuffer = inputs
    }

    private initializeNeurons(
        activationFunction: ActFunc,
        neuronCount: number,
        inputBufferSize: number
    ): Neuron[] {
        let arrayOfNeurons: Neuron[] = []
        for (let i = 0; i < neuronCount; i++) {
            let arrayOfWeights: number[] = []
            for (let j = 0; j < inputBufferSize; j++) {
                arrayOfWeights[j] = this.generateSmallNonzeroValue()
            }
            arrayOfNeurons[i] = new Neuron(
                activationFunction,
                arrayOfWeights,
                this.generateSmallNonzeroValue()
            )
        }

        return arrayOfNeurons
    }

    private generateSmallNonzeroValue() {
        const numberNearZero = 2 * Math.random() - 1
        const nonZeroNumber = numberNearZero === 0 ? 1 : numberNearZero
        const numberEvenNearerToZero = 3 * (nonZeroNumber ** 3 + nonZeroNumber / 3) / 4
        return numberEvenNearerToZero
    }
}

