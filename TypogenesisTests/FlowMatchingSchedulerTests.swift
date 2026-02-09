import Testing
import Foundation
@testable import Typogenesis

@Suite("FlowMatchingScheduler Tests")
struct FlowMatchingSchedulerTests {

    // MARK: - Initialization

    @Test("Scheduler creates correct number of timesteps")
    func testTimestepCount() {
        let scheduler = FlowMatchingScheduler(numSteps: 50)
        #expect(scheduler.timesteps.count == 51, "Should have numSteps + 1 timesteps")
        #expect(scheduler.numSteps == 50)
    }

    @Test("Scheduler with 10 steps has 11 timesteps")
    func testSmallStepCount() {
        let scheduler = FlowMatchingScheduler(numSteps: 10)
        #expect(scheduler.timesteps.count == 11)
    }

    @Test("Scheduler with 1 step works")
    func testSingleStep() {
        let scheduler = FlowMatchingScheduler(numSteps: 1)
        #expect(scheduler.timesteps.count == 2)
        #expect(scheduler.timesteps[0] == 1.0)
        #expect(scheduler.timesteps[1] == 0.0)
    }

    // MARK: - Timestep Values

    @Test("First timestep is 1.0 (pure noise)")
    func testFirstTimestep() {
        let scheduler = FlowMatchingScheduler(numSteps: 50)
        #expect(scheduler.timesteps[0] == 1.0)
    }

    @Test("Last timestep is 0.0 (clean data)")
    func testLastTimestep() {
        let scheduler = FlowMatchingScheduler(numSteps: 50)
        #expect(scheduler.timesteps[50] == 0.0)
    }

    @Test("Timesteps are monotonically decreasing")
    func testMonotonicity() {
        let scheduler = FlowMatchingScheduler(numSteps: 50)
        for i in 0..<scheduler.numSteps {
            #expect(scheduler.timesteps[i] > scheduler.timesteps[i + 1],
                    "Timestep \(i) (\(scheduler.timesteps[i])) should be > timestep \(i+1) (\(scheduler.timesteps[i+1]))")
        }
    }

    @Test("Timesteps are evenly spaced")
    func testEvenSpacing() {
        let scheduler = FlowMatchingScheduler(numSteps: 10)
        let expectedStep: Float = 0.1
        for i in 0..<scheduler.numSteps {
            let diff = scheduler.timesteps[i] - scheduler.timesteps[i + 1]
            #expect(abs(diff - expectedStep) < 1e-5,
                    "Step \(i) diff = \(diff), expected \(expectedStep)")
        }
    }

    // MARK: - Euler Step

    @Test("Euler step with zero velocity leaves sample unchanged")
    func testZeroVelocity() {
        let scheduler = FlowMatchingScheduler(numSteps: 10)
        let sample: [Float] = [1.0, 2.0, 3.0, 4.0]
        let velocity: [Float] = [0.0, 0.0, 0.0, 0.0]

        let result = scheduler.step(velocity: velocity, stepIndex: 0, sample: sample)

        for i in 0..<sample.count {
            #expect(abs(result.prevSample[i] - sample[i]) < 1e-5,
                    "With zero velocity, sample should be unchanged at index \(i)")
        }
    }

    @Test("Euler step moves sample in correct direction")
    func testStepDirection() {
        let scheduler = FlowMatchingScheduler(numSteps: 10)
        let sample: [Float] = [0.0, 0.0]
        let velocity: [Float] = [1.0, -1.0]

        // dt = timesteps[0] - timesteps[1] = 1.0 - 0.9 = 0.1
        // prevSample = sample - dt * velocity = [0 - 0.1*1, 0 - 0.1*(-1)] = [-0.1, 0.1]
        let result = scheduler.step(velocity: velocity, stepIndex: 0, sample: sample)

        #expect(abs(result.prevSample[0] - (-0.1)) < 1e-5, "Expected -0.1, got \(result.prevSample[0])")
        #expect(abs(result.prevSample[1] - 0.1) < 1e-5, "Expected 0.1, got \(result.prevSample[1])")
    }

    @Test("Euler step computes correct predX0")
    func testPredX0() {
        let scheduler = FlowMatchingScheduler(numSteps: 10)
        let sample: [Float] = [1.0, 2.0]
        let velocity: [Float] = [0.5, 1.0]

        // At step 0: t_current = 1.0
        // predX0 = sample - t_current * velocity = [1 - 1*0.5, 2 - 1*1] = [0.5, 1.0]
        let result = scheduler.step(velocity: velocity, stepIndex: 0, sample: sample)

        #expect(abs(result.predX0[0] - 0.5) < 1e-5)
        #expect(abs(result.predX0[1] - 1.0) < 1e-5)
    }

    @Test("Step at different indices uses correct dt")
    func testDifferentStepIndices() {
        let scheduler = FlowMatchingScheduler(numSteps: 5)
        // timesteps: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        // dt is always 0.2 for this scheduler
        let sample: [Float] = [1.0]
        let velocity: [Float] = [1.0]

        // Step 0: prevSample = 1.0 - 0.2 * 1.0 = 0.8
        let r0 = scheduler.step(velocity: velocity, stepIndex: 0, sample: sample)
        #expect(abs(r0.prevSample[0] - 0.8) < 1e-5)

        // Step 4: dt still 0.2, prevSample = 1.0 - 0.2 * 1.0 = 0.8
        let r4 = scheduler.step(velocity: velocity, stepIndex: 4, sample: sample)
        #expect(abs(r4.prevSample[0] - 0.8) < 1e-5)
    }

    @Test("Full denoising loop with constant velocity converges")
    func testFullLoop() {
        let scheduler = FlowMatchingScheduler(numSteps: 10)
        // With constant velocity = 1.0, sample starts at 0.0
        // After full loop: sample = 0.0 - sum(dt_i * 1.0) = 0.0 - 1.0 = -1.0
        var sample: [Float] = [0.0]
        let velocity: [Float] = [1.0]

        for i in 0..<scheduler.numSteps {
            let result = scheduler.step(velocity: velocity, stepIndex: i, sample: sample)
            sample = result.prevSample
        }

        #expect(abs(sample[0] - (-1.0)) < 1e-4,
                "Full loop should move sample by -1.0, got \(sample[0])")
    }

    // MARK: - Output Shape Consistency

    @Test("Step output has same length as input")
    func testOutputShape() {
        let scheduler = FlowMatchingScheduler(numSteps: 10)
        let sizes = [1, 64, 4096]

        for size in sizes {
            let sample = [Float](repeating: 0, count: size)
            let velocity = [Float](repeating: 1.0, count: size)
            let result = scheduler.step(velocity: velocity, stepIndex: 0, sample: sample)
            #expect(result.prevSample.count == size)
            #expect(result.predX0.count == size)
        }
    }
}
