import { describe, it, expect } from 'vitest'
import Neuron from "./neuron"

describe("Neuron", function() {
    describe("constructor", function () {
        it("accepts a callback for an activation function", function() {
            const neuron = new Neuron(() => {
                return 0.7
            }, [], 4.3)

            expect(typeof neuron.actFunc).toBe("function")
            expect(neuron.actFunc(Math.random())).toBe(0.7)
        })

        it("accepts an array of float values between 0 and 1 for initial weights", function() {
            const initialWeights = [1, 0.5, 0]
            const neuron = new Neuron((n: number) => n, initialWeights, 4.3)
            expect(neuron.weights).toEqual(initialWeights)
        })

        it("accepts a float value for bias", function() {
            const neuron = new Neuron((n: number) => n, [], 4.3)
            expect(neuron.bias).toEqual(4.3)
        })

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
            const identityFunction = (weightedSum: number): number => { return weightedSum }

            it("produces a linear combination of weights and inputs", function() {
                const inputs: number[] = [2.0, 3.0, 5.0]
                const weights: number[] = [7.0, 11.0, 13.0]
                const bias: number = 17

                const neuron: Neuron = new Neuron(identityFunction, weights, bias)
                expect(neuron.eval(inputs)).toBe(129)
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
            const stepFunction = (weightedSum: number): number => {
                return weightedSum >= 0 ? 1 : 0
            }

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

        describe("When activation func returns zero", function() {
            it("returns 0 always", function() {
                const inputs = [Math.random(), Math.random(), Math.random()]
                const weights = [Math.random(), Math.random(), Math.random()]
                const bias = Math.random()
                const neuron: Neuron = new Neuron(() => 0, weights, bias)
                expect(neuron.eval(inputs)).toBe(0)
            })
        })
    })
})
