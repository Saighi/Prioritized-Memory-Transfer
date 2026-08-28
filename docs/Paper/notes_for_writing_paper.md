"For any downstream computation that is Lipschitz-continuous over the relevant cortical memory manifold, the representational reconstruction certificate immediately induces a uniform bound on functional error."

"le modèle traite le cortex comme une mémoire associative afin d’isoler le principe de consolidation; dans un système où les représentations sont transformées, le même raisonnement s’appliquerait à l’erreur dans l’espace cortical cible, ou plus généralement à une distance fonctionnellement pertinente."

"We model cortex as an associative memory to make the consolidation objective explicit. This should not be interpreted as requiring hippocampal and cortical representations to be identical. More generally, cortical consolidation may involve a transformation of hippocampal content into a distinct cortical code; in that case, the same minimax principle applies to the worst-case discrepancy relative to the appropriate cortical target representation."

"Once the maximally discrepant hippocampal state has been identified, cortical plasticity follows the locally steepest direction for reducing the resulting worst-case reconstruction bound."

"the rule is locally optimal in the sense that it produces the greatest first-order decrease of the worst-case deficit per unit synaptic change"

"We abstracted hippocampal encoding as the addition of a pattern-separated trace to the teacher’s associative memory. The upstream mechanism producing separated traces was not modeled."

"Under the simple-worst-case-mode assumption, the maximizing teacher direction is locally unique up to sign. The two signs produce the same weight derivative. Consequently, \(\mathcal D_{\max}\) is locally differentiable and the cortical plasticity rule implements gradient descent on \(\mathcal D_{\max}\)."

"As a whole, the architecture identifies the teacher-supported memory with the largest settled student deficit and then reduces that worst-case deficit. Among all student-weight changes of the same small magnitude, the resulting plasticity event produces the greatest first-order reduction in this deficit."

"Simple-leading-eigenvalue assumption. We assume that the largest eigenvalue of \(A_S\) has multiplicity one:
\[
\lambda_1(A_S)>\lambda_2(A_S).
\]The maximizing teacher states are therefore the antipodal pair \(x_T^*\) and \(-x_T^*\), which produce the same weight gradient. Exact multidirectional degeneracies are outside the scope of this result."

"At complete transfer, all discrepancy eigenvalues may coincide at zero; this degeneracy is harmless because the maximal deficit and plasticity signal already vanish."

"Then summarize the positive replay gain, prioritization, self-extinction, random-versus-oracle efficiency, and rectification."

"Existing theories commonly prescribe replay priority, generate replay through autonomous memory dynamics, or explain how replay trains a recipient network. Here, these functions arise within a single closed-loop mechanism: recipient reconstruction deficits reshape source dynamics, preferentially exposing unconsolidated content, while learning progressively removes the signal that caused its reactivation."

The model concentrates reactivation on the current principal deficit direction and reallocates reactivation as the discrepancy spectrum changes.

"A related mechanism was proposed by Wang, Poe, and Zochowski (2008), whose spiking-network model showed that the progressive formation of a cortical memory trace could recruit inhibitory feedback that shortened reactivation of the corresponding hippocampal trace. This provided an early demonstration of autonomous, familiarity-dependent replay control. However, replay priority in their model arose from cortical attractor strength and diffuse feedback inhibition, rather than from an explicit comparison between source activity and the receiver’s reconstruction. Consequently, the mechanism was not connected to a normative consolidation objective and did not express replay priority as a representation-specific residual. Our model recasts this general circuit motif as predictive cancellation: cortical activity provides a negative image that selectively removes already-represented components from hippocampal dynamics, leaving the recipient-relative mismatch to drive further replay and learning."
