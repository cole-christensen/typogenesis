import Foundation
import Accelerate

/// Pure Swift implementation of the flow-matching scheduler for diffusion sampling.
///
/// Ports the Python `FlowMatchingScheduler` from `noise_schedule.py`.
/// The scheduler manages Euler ODE integration steps that transform noise → data:
///   x_{t-dt} = x_t - dt * v(x_t, t)
///
/// Timesteps go from 1.0 (pure noise) to 0.0 (clean data) in equal steps.
struct FlowMatchingScheduler: Sendable {

    /// Timestep values from 1.0 → 0.0 (length = numSteps + 1)
    let timesteps: [Float]

    /// Number of denoising steps
    let numSteps: Int

    /// Result of a single scheduler step
    struct StepResult {
        /// The denoised sample after the Euler step
        let prevSample: [Float]
        /// Estimated clean image at t=0 (for visualization)
        let predX0: [Float]
    }

    /// Create a scheduler with the given number of inference steps.
    ///
    /// - Parameter numSteps: Number of denoising steps (default: 50).
    init(numSteps: Int = 50) {
        precondition(numSteps > 0, "numSteps must be positive")
        self.numSteps = numSteps

        // Linear schedule from 1.0 (noise) to 0.0 (data)
        var ts = [Float](repeating: 0, count: numSteps + 1)
        for i in 0...numSteps {
            ts[i] = 1.0 - Float(i) / Float(numSteps)
        }
        self.timesteps = ts
    }

    /// Perform one Euler step of the ODE.
    ///
    /// - Parameters:
    ///   - velocity: Predicted velocity field v(x_t, t), flattened array.
    ///   - stepIndex: Current step index (0-based, 0 to numSteps-1).
    ///   - sample: Current noisy sample x_t, same shape as velocity.
    /// - Returns: StepResult with the updated sample and predicted x_0.
    func step(velocity: [Float], stepIndex: Int, sample: [Float]) -> StepResult {
        precondition(stepIndex >= 0 && stepIndex < numSteps,
                     "stepIndex \(stepIndex) out of range [0, \(numSteps))")
        precondition(velocity.count == sample.count,
                     "velocity and sample must have the same length")

        let tCurrent = timesteps[stepIndex]
        let tNext = timesteps[stepIndex + 1]
        let dt = tCurrent - tNext  // Positive since going 1 → 0

        let count = sample.count

        // prevSample = sample - dt * velocity  (Euler step)
        var prevSample = [Float](repeating: 0, count: count)
        var negDt = -dt
        vDSP_vsma(velocity, 1, &negDt, sample, 1, &prevSample, 1, vDSP_Length(count))

        // predX0 = sample - tCurrent * velocity  (estimated clean image)
        var predX0 = [Float](repeating: 0, count: count)
        var negT = -tCurrent
        vDSP_vsma(velocity, 1, &negT, sample, 1, &predX0, 1, vDSP_Length(count))

        return StepResult(prevSample: prevSample, predX0: predX0)
    }
}
