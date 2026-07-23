Considering the paper from shin et Jadhav showing that PCF inhibit CA1 activity we can have a neat parralel between the model behavior and the neuroscience.

Canonical predictive coding suppresses residual-error activity while supporting matching representations. In contrast, prefrontal ripples appear capable of suppressing hippocampal replay representations themselves. Our model proposes a possible function for this noncanonical interaction: cortical feedback may subtract content already represented by the receiver, allowing poorly shared hippocampal components to dominate subsequent replay.

The sign reversal converts an ordinary predictive-coding correction signal into a predictive-cancellation signal, and then feeds the uncancelled residual back into the dynamics that generate replay.

so in my model the student is top in the hierachy compared to the teacher, contrary to what has been said. 

reversing the sign of precision turn a classical PC interpretation into a corolarry discharge interpretation where top activity
directly inhibate bottom activity when predictible.

it is very important to well explain the difference between inhibiting error neurons which themselve inhibit representation neurons which push for matching activity, and inhibiting representation neurons directly. my model is the second case during sleep.

we have corollary-discharge-like predictive cancellation

a way to put it is :

"Predictive coding and corollary-discharge circuits use learned predictions to cancel expected activity, leaving a residual that signals unpredicted input. We asked whether the same elementary computation could solve the replay-selection problem between memory systems. In our model, a sleep-gated reversal of predictive-coding error feedback causes the receiving population’s reconstruction to act as a negative image at the source. Shared activity is cancelled, whereas source-supported activity that the receiver cannot reconstruct survives as a residual. By feeding this residual back into the source dynamics, the model converts predictive cancellation into prioritized replay."

my work show that "predictive cancellation can do more than distinguish expected from unexpected sensory input: when applied recurrently between memory populations, it generates an adaptive and self-terminating replay curriculum."

but this need to take a step aside classical predictive coding.

We have three more papers shown this kind of predictive cancelation in the neocortex that may be usefull as citation:

Garner and Keller establish a learned cross-population predictive-coding interaction but interpret its suppression primarily in terms of prediction-error neurons. Schneider’s work provides a closer circuit analogy to my mechanism: activity in one population recruits local inhibition that suppresses matching representational activity in another. I extend this predictive-cancellation motif to interacting memory populations, where the uncancelled residual is fed back into the source dynamics and thereby determines what is replayed and transferred.

A last argument in shin et Jadhav we see the PCF inhibiting direcly CA1 neurons pyramidal soma activity.
These neurons are not usually seen as error neurons. For two reasons, first many PC interpretation put the error computation that 
has to be inhibited in the dendrites. Second, they don't have the profile of error neurons.

what is comming back from the litterature in general is that Long-range cortical projections can select among local inhibitory and disinhibitory microcircuits, allowing learned top-down signals to either suppress predictable representations or amplify unexpected ones.

This switch between amplification and pure inhibition is my sign reversal.

Predictive negative-image circuits provide a biological precedent for receiver-driven replay control. In sensory cortex, learned representations in frontal or cross-modal populations can recruit inhibition that cancels matching sensory activity. We propose that, during offline memory communication, the same circuit motif may act recurrently on the generating population: a receiver’s reconstruction suppresses already-shared source activity, while the uncancelled residual biases subsequent replay and transfer.

so to conclude, there is two different interpretation possible of top down inhibition.
either it focus on error units and theirfore only filter the bottom up signal in an inhibitory way but can push for alignment 
through top down influence, this is classical PC coding interpretation.

Or it directly modify the activity of representation neurons, which exert a similar filter from bottom to up but 
can bias the trajectory taken by the bottom representation neuron in an inverse direction.

this is typically what my network do, as their are recurrent synapses in my bottom population (the teacher), by creating an
negative image we push it toward exploring other memories. this can't be obtain by classical alignment PC.