# Brief for the first empirical Results figure

## Purpose

This is the first large empirical figure in the Results. It should establish the paper's main phenomenon before the normative objective and most of the analysis are introduced.

The central result is that coupled intrinsic dynamics can redistribute the complete memory content of a frozen hippocampal-like teacher into an initially empty cortical-like student. No external memory is presented during this phase. Teacher activity spontaneously explores stored content, the student follows and learns, the student's memory space progressively acquires the teacher's memory space, and the signals responsible for reactivation and plasticity disappear when transfer is complete.

Prioritization should already be visible in this trajectory, but this figure is not the controlled prioritization result. Recipient-relative priority, familiarity manipulations, and comparisons with random or oracle selection will be developed in later figures.

Provisional figure title:

> **Spontaneous intrinsic dynamics redistribute a complete memory representation**

## Simulation concept

Store three distinct MNIST images in the teacher network. Use a network large enough to represent the chosen image resolution directly. The student begins without these memories. Freeze the teacher weights, remove external input, activate the teacher-student coupling, and allow student plasticity to operate during the spontaneous trajectory.

The three images should be visually distinct and linearly independent. Their selection must be reported transparently. The figure may use a representative run, but it should not depend on a trajectory chosen solely because it looks attractive. Later figures or Supporting Information can establish seed robustness.

The main figure should use one internally consistent model condition. In particular, it should not mix a linear trajectory in some panels with a rectified trajectory in others. The choice between the linear and rectified model should respect the claim made in the text:

- The linear model supports the exact memory-subspace and kernel-inclusion interpretation, but its dreams may be signed mixtures rather than clean individual digits.
- The rectified model can produce more recognizable, memory-like dreams, but the figure and caption must not silently treat its nonlinear state space as the same object as the linear memory subspace.

Inspect the existing MNIST dream simulations and choose the version that gives a readable figure without overstating what the dynamics represent. If the linear condition is used, mixtures and sign changes are part of the result and should be displayed honestly.

## Proposed composition

Use four labeled panels. Panel A should dominate the layout. Large multi-panel figures are acceptable, but the reader should still see one continuous story rather than four unrelated diagnostics.

### Panel A. Spontaneous trajectory through stored memory space

Show the teacher trajectory projected into the three-dimensional space spanned by the stored MNIST images. Use an orthonormal representation of this space so that distances and directions in the plot are meaningful.

Include:

- the teacher trajectory, colored continuously by time;
- the student trajectory, if it remains legible, using a quieter visual treatment;
- the locations of the three stored images;
- small thumbnails of the stored MNIST images near their locations;
- four marked trajectory times, labelled consistently as (t_1,t_2,t_3,t_4).

The stored-image thumbnails should orient the reader without covering the trajectory. They should remain secondary to the state-space motion.

The panel should make clear that this is a projection into the exact three-dimensional teacher memory space, not a three-unit network. If the linear model is used, do not draw the three memories as isolated attractors. They span a continuous memory space that also contains mixtures.

### Panel B. Four dream snapshots along the trajectory

Show four snapshots corresponding exactly to (t_1,t_2,t_3,t_4) in panel A. Link them through matching colors or markers.

The preferred arrangement has two rows:

- teacher activity, interpreted as the current dream or reactivated content;
- simultaneous student activity, showing the recipient's current reconstruction.

This filmstrip should let a reader see memory content moving through the coupled system. Preserve negative activity if it is present in the linear model. Do not clip or rectify it merely to make the images look more like MNIST digits. Any display transformation needed for visibility must be stated in the caption.

Choose the four times by a reproducible rule tied to transfer progress or major stages of the trajectory. Do not select frames only because they resemble the stored digits particularly well.

### Panel C. The student acquires the teacher memory space

Show the recipient's error spectrum restricted to the teacher memory space. If (U_T) is an orthonormal basis of the teacher memory space and (M_S(t)=I-W_S(t)), the relevant operator is

\[
U_T^\top M_S(t)^\top M_S(t)U_T.
\]

Plot its three eigenvalues through time. Mark (t_1,t_2,t_3,t_4) on the same axis.

The eigenvalues should fall toward zero as the student acquires each independent teacher-supported direction. Their joint disappearance is the direct numerical statement of complete redistribution:

\[
M_S U_T=0
\quad\Longleftrightarrow\quad
\ker M_T\subseteq\ker M_S.
\]

Call this the recipient error spectrum on the teacher memory space. Avoid a thresholded plot of "kernel dimension," since exact numerical zeros depend on an arbitrary tolerance. Also avoid using convergence of (W_S) toward (W_T) as the transfer criterion. The two networks may support the same memories without having identical weights.

### Panel D. Completion removes the reactivation and learning signals

Show the quantities needed to close the trajectory:

- total teacher-space residual, such as \(\|M_SU_T\|_F\);
- the magnitude of the teacher-side selection drive;
- the magnitude of student plasticity.

Use aligned traces or compact stacked axes rather than forcing quantities with different units onto an unclear shared scale. Mark the four snapshot times again.

The result should be visually unambiguous. Full transfer coincides with the disappearance of the discrepancy-driven selection signal and the local plasticity signal. Teacher activity may continue to diffuse because noise and normalization remain present, so the claim is that structured selection and learning extinguish themselves, not necessarily that all neuronal activity stops.

If space permits, include a small check that teacher activity remains concentrated on the teacher memory space. This rules out apparent transfer caused by the teacher chasing off-manifold noise.

## Visual organization

A natural arrangement is:

```text
┌──────────────────────────────┬──────────────────────┐
│ A. 3D trajectory, memories   │ C. restricted error │
│    and four marked times     │    spectrum          │
├──────────────────────────────┼──────────────────────┤
│ B. four paired dream frames  │ D. transfer and      │
│    teacher above student     │    self-extinction   │
└──────────────────────────────┴──────────────────────┘
```

Panel A should receive the most space. Panels C and D should share a time axis when practical. Use the same colors for time, snapshot markers, and corresponding vertical lines throughout the figure. Keep legends compact and avoid large titles inside individual panels.

## Text surrounding the figure

The subsection should begin with the empirical question, not the objective:

> We first asked whether coupled intrinsic activity could redistribute an entire memory representation without external presentation of the stored content. We initialized a frozen teacher with three MNIST memories and an empty student, removed external input, and allowed only the coupled state dynamics and local student plasticity to operate.

Before the figure, the only mathematical reminder needed is that a linear network stores the states in \(\ker M\), and complete transfer therefore requires \(M_SU_T=0\). Define the restricted error spectrum only as needed to read panel C.

After the figure, state the empirical conclusion:

> Teacher activity spontaneously moved through its stored memory space while the student followed and learned. Across the same trajectory, every student error mode supported by the teacher approached zero. The student therefore acquired the complete teacher memory space rather than only the particular states sampled at the displayed times. Once transfer was complete, the selection and plasticity signals disappeared.

Do not claim from this figure alone that transfer is faster than random reactivation. That claim requires the later matched comparison. Likewise, describe prioritization as visible in the trajectory, then test it directly in the subsequent recipient-deficit experiments.

## Place in the Results

The Results should proceed in this order:

1. **Spontaneous intrinsic dynamics redistribute the complete teacher memory space.** This figure and minimal descriptive mathematics.
2. **A worst-case recipient objective explains the transfer dynamics.** Introduce the settled deficit, the worst-case objective, its spectral form, the coupling-sign derivation, local descent, and the completion condition. Keep full proofs in the appendix.
3. **Reactivation follows the largest recipient deficit and changes as learning proceeds.** Quantify prioritization, switching, self-extinction, and recipient familiarity.
4. **Autonomous prioritization improves transfer efficiency.** Compare autonomous dynamics with the greedy oracle and random reactivation under a matched protocol.
5. Present waking-strength manipulations, robustness tests, and the rectified extension in the order supported by the final data.

The first figure must make the phenomenon worth explaining. The following mathematics then tells the reader why these spontaneous dynamics select missing content, transfer it, and stop.
