import { describe, it, expect } from 'vitest'
import Neuron from "./neuron"

describe("Neuron", function() {
    describe("constructor", function () {
        it("accepts a callback for an activation function", function() {
            const neuron = new Neuron(() => {
                return 0.7
            }, [], 4.3)

            expect(typeof neuron.actFunc).toBe("function")
            expect(neuron.actFunc()).toBe(0.7)
        });

        it("accepts an array of float values between 0 and 1 for initial weights", function() {
            const initialWeights = [1, 0.5, 0]
            const neuron = new Neuron(() => {
            }, initialWeights, 4.3)
            expect(neuron.weights).toEqual(initialWeights)
        });

        it("accepts a float value for bias", function() {
            const neuron = new Neuron(() => {}, [], 4.3)
            expect(neuron.bias).toEqual(4.3)
        });

        // defaults for weights and bias if none provided??
    })

    describe("#eval", function() {

        it("returns value of the bias weight when all inputs are 0, regardless of weights", function() {
            const inputs: number[] = [0, 0]
            const weights: number[] = [3.0, 1.1]
            const bias: number = 52

            const neuron: Neuron = new Neuron((x: number) => {return x}, weights, bias)
            expect(neuron.eval(inputs)).toBe(bias)
        })

        it("returns value of the bias when all the weights are 0, regardless of input", function() {
            const inputs: number[] = [999, -8765]
            const weights: number[] = [0, 0]
            const bias: number = 3

            const neuron: Neuron = new Neuron((x: number) => {return x}, weights, bias)
            expect(neuron.eval(inputs)).toBe(bias)
        })

        describe("assuming the activation function is the identity function", function() {
            const identityFunction = (weightedSum: number) :number => { return weightedSum }

            it("returns length of input array when all the inputs and weights are 1.0 and bias is 0.0", function() {
                // Since the product of each input-and-weight pair will be 1.0, then each pair will contribute 1.0 to the overall sum.
                // Therefore, this is a special case where the sum will simply be the count of all pairs.
                // In the end, this might be too clever, but it does validate the contribution of each pair.
                const inputs: number[] = [1.0, 1.0, 1.0]
                const weights: number[] = [1.0, 1.0, 1.0]
                const bias: number = 0

                const neuron: Neuron = new Neuron(identityFunction, weights, bias)
                expect(neuron.eval(inputs)).toBe(inputs.length)
            })

            it("returns 0.5 when all the inputs are 1.0, and any single weight is 0.5 while the rest are 0 and the bias is 0", function() {
                const inputs: number[] = [1.0, 1.0, 1.0]
                const weights: number[] = [0, 0.5, 0]
                const bias: number = 0

                const neuron: Neuron = new Neuron(identityFunction, weights, bias)
                expect(neuron.eval(inputs)).toBe(0.5)
            })

            it("returns 0.5 when all the weights are 1.0, and any single input is 0.5 while the rest are 0 and the bias is 0", function() {
                const inputs: number[] = [0, 0.5, 0]
                const weights: number[] = [1.0, 1.0, 1.0]
                const bias: number = 0

                const neuron: Neuron = new Neuron(identityFunction, weights, bias)
                expect(neuron.eval(inputs)).toBe(0.5)
            })

            it("returns sum of all the inputs weighted by the neuron weights plus the bias", function() {
                const inputs: number[] = [0.25, 6]
                const weights: number[] = [100, -3]
                const bias: number = 12

                const neuron: Neuron = new Neuron(identityFunction, weights, bias)
                expect(neuron.eval(inputs)).toBe(19)
            })
        })

        describe("assuming the activation function is the unit step function", function() {
            const stepFunction = (weightedSum: number) :number => { return weightedSum >= 0 ? 1 : 0 }

            it("returns 1.0 when all the inputs and weights are 1.0 and bias is 0.0", function() {
                const inputs: number[] = [1.0, 1.0, 1.0]
                const weights: number[] = [1.0, 1.0, 1.0]
                const bias: number = 0

                const neuron: Neuron = new Neuron(stepFunction, weights, bias)
                expect(neuron.eval(inputs)).toBe(1.0)
            })

            it("returns 1.0 when all the inputs are 1.0, and any single weight is 0.5 while the rest are 0 and the bias is 0", function() {
                const inputs: number[] = [1.0, 1.0, 1.0]
                const weights: number[] = [0, 0.5, 0]
                const bias: number = 0

                const neuron: Neuron = new Neuron(stepFunction, weights, bias)
                expect(neuron.eval(inputs)).toBe(1.0)
            })

            it("returns 1.0 when all the weights are 1.0, and any single input is 0.5 while the rest are 0 and the bias is 0", function() {
                const inputs: number[] = [0, 0.5, 0]
                const weights: number[] = [1.0, 1.0, 1.0]
                const bias: number = 0

                const neuron: Neuron = new Neuron(stepFunction, weights, bias)
                expect(neuron.eval(inputs)).toBe(1.0)
            })

            it("returns 1.0 when weighted sum + bias is positive", function() {
                const inputs: number[] = [0.25, 6]
                const weights: number[] = [100, -3]
                const bias: number = 12

                const neuron: Neuron = new Neuron(stepFunction, weights, bias)
                expect(neuron.eval(inputs)).toBe(1.0)
            })
        })
    })
});
