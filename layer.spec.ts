import { describe, it, expect } from 'vitest'
import Layer from './layer'
import Neuron from "./neuron"

describe("Layer", function() {
    describe("constructor", function () {
        it("accepts an integer for the number of neurons", function() {
            const layer: Layer = new Layer((n: number) => n, 2, 4)
            expect(layer.neurons.length).toBe(2)
        })

        it("accepts an integer for the number of inputs", function() {
            const layer: Layer = new Layer((n: number) => n, 2, 4)
            expect(layer.inputBufferSize).toBe(4)
        })

        it("initializes with an empty input buffer", function() {
            const layer: Layer = new Layer((n: number) => n, 2, 4)
            expect(layer.inputBuffer).toEqual([])
        })

        it("initializes each neuron with a set of weights of count equal to the inputBufferSize", function () {
            const layer: Layer = new Layer((n: number) => n, 2, 4)

            layer.neurons.forEach((neuron) => {
                expect(neuron.weights.length).toEqual(4)
            })
        })

        it("initializes each neuron with a random set of small non-zero weights", function () {
            const layer: Layer = new Layer((n: number) => n, 2, 4)

            layer.neurons.forEach( (neuron: Neuron) => {
                neuron.weights.forEach((weight: number) => {
                    expect(weight).toBeGreaterThanOrEqual(-1.0)
                    expect(weight).toBeLessThanOrEqual(1.0)
                    expect(weight).not.toEqual(0)
                })
            })
        })

        it("initializes each neuron with a random small non-zero bias", function () {
            const layer: Layer = new Layer((n: number) => n, 2, 4)

            layer.neurons.forEach( (neuron: Neuron) => {
                expect(neuron.bias).toBeGreaterThanOrEqual(-1.0)
                expect(neuron.bias).toBeLessThanOrEqual(1.0)
                expect(neuron.bias).not.toEqual(0)
            })
        })
    })

    describe("#applyInputs", function() {
        it("accepts an array of inputs matching the inputBufferSize", function() {
            const layer: Layer = new Layer((n: number) => n, 2, 4)
            expect(() => { layer.applyInputs([4,3,2,1]) }).not.toThrow()
        })

        it("throws an error when the argument length does not match the inputBufferSize", function() {
            const layer: Layer = new Layer((n: number) => n, 2, 4)
            expect(() => { layer.applyInputs([1]) }).toThrow()
        })

        it("sets the inputBuffer to the passed in array", function() {
            const layer: Layer = new Layer((n: number) => n, 2, 4)
            layer.applyInputs([4,3,2,1])
            expect(layer.inputBuffer).toEqual([4,3,2,1])
        })
    })

    // describe("#eval", function() {
    //     it("")
    // })
})
