# 1. Introduction

*Draft v1 — sections 1.1 to 1.4. Section 1.5 (Aims and Objectives) awaits your wording.*
*Word count: §1.1–1.4 = 1,462 words.*

---

## 1.1 Background and motivation

Major depressive disorder is among the most prevalent mental health conditions worldwide,
and is characterised by persistent low mood affecting individuals across all age groups [1].
Diagnosis remains largely dependent on clinical interview and self-report instruments, a
process that is time-consuming, requires trained clinicians, and is subject to variation
between assessors. Abnormal facial behaviour during clinical interaction has long been
identified as a diagnostic cue [2], which has motivated a substantial body of work seeking
to estimate depression severity automatically from facial video.

The computational task addressed by that literature, and by this dissertation, is a
regression problem. Given a video of a participant's face, a model predicts that
participant's score on the Beck Depression Inventory-II, a self-report questionnaire whose
values range from 0 to 63 and which is conventionally partitioned into four severity bands:
minimal (0–13), mild (14–19), moderate (20–28), and severe (29–63) [3]. Two public corpora
released for the Audio/Visual Emotion Challenge, AVEC 2013 [3] and AVEC 2014 [4], have
become the standard benchmarks, and reported performance is almost universally expressed
as mean absolute error and root mean square error against the held-out test partition.

This is a domain in which the reliability of a reported result carries unusual weight.
A system that estimates a clinical severity score from a person's face is, if deployed,
consequential in a way that a general-purpose image classifier is not; and the same
properties that make the task attractive to researchers — small, well-defined public
corpora with a single scalar target — also make it easy to produce a strong number that
does not generalise. Independent replication is therefore not merely a methodological
courtesy in this field but a precondition for taking any reported figure seriously.
This dissertation reports such a replication, of a recent and well-cited method, together
with an assessment of what the reported result depends upon.

## 1.2 Vision-based estimation of depression severity

Early approaches to this task combined hand-crafted descriptors of facial appearance and
motion with conventional regressors, and were largely superseded once convolutional
networks were applied to the problem directly. Zhu et al. [5] encoded facial appearance and
dynamics using two parallel deep networks, establishing that learned representations
outperformed engineered ones on the AVEC corpora. Zhou et al. [6] subsequently pursued
interpretability alongside accuracy, learning representations whose activations could be
related to specific facial regions.

Because depression manifests in behaviour that unfolds over time rather than in any single
expression, attention within the field moved towards architectures that model temporal
structure explicitly. Jazaery and Guo [7] extracted feature sequences using three-dimensional
convolution and aggregated them with a recurrent network. De Melo et al. [8] instead
combined global and local three-dimensional networks, processing the whole face and the
periocular region in parallel and dispensing with a recurrent stage entirely. The same
authors later proposed a multiscale spatiotemporal network [9], and a decomposed variant
[10] that reduced computational cost while preserving temporal modelling capacity; the
latter provides the MDN baselines against which subsequent work is commonly compared.

A parallel line of work has questioned the standard formulation of the output. Because the
Beck Depression Inventory-II is an ordinal instrument with clinically meaningful bands and
a markedly skewed population distribution, treating its score as an unconstrained real
number discards structure. De Melo et al. [11] and Zhou et al. [12] therefore recast the
problem as label distribution learning, predicting a distribution over possible scores
rather than a point estimate, with the latter combining distribution learning and metric
learning in a joint objective.

Two observations about this lineage are relevant to the present work. The first is that
reported errors have converged into a narrow band: methods published between 2018 and 2024
report mean absolute errors on AVEC 2014 largely between approximately 6 and 8, so that
successive contributions are separated by margins of a few tenths of a point. The second is
that essentially all of these results are reported as a single figure per method per corpus,
without confidence intervals, repeated runs, or any other indication of how much of the
margin is attributable to the method rather than to the training run. Section 1.4 returns
to why this matters.

## 1.3 Attention mechanisms and the method under replication

Attention mechanisms were introduced to visual recognition to allow a network to weight
features adaptively rather than uniformly. Xu et al. [13] applied soft attention to image
captioning; Woo et al. [14] proposed the convolutional block attention module, which
computes channel and spatial attention sequentially within a residual block; and Wang et al.
[15] introduced non-local operations that relate features across all positions, at
considerable memory cost.

The method replicated in this dissertation, the Spatial–Temporal Attention Depression
Recognition Network of Pan et al. [16], applies this idea to the depression regression task.
Its stated motivation is a twofold criticism of prior work: that three-dimensional
convolutional networks dilute spatial information through translation-invariant operations
and so weight all facial regions equally, and that existing attention mechanisms either mix
spatial and temporal information during extraction or operate only in two dimensions. The
proposed module replaces the middle convolution of each residual bottleneck with two
parallel branches. A spatial branch pools features along the temporal axis to determine
which locations are consistently informative; a temporal branch pools along the spatial
axes to determine which frames are informative. The two resulting attention vectors are
combined multiplicatively into a single spatial–temporal vector, which is then applied to
the summed branch outputs — a design the authors term attention vector-wise fusion, and
which they distinguish from the simpler alternative of summing the two branches directly.

The reported results are mean absolute error and root mean square error of 6.15 and 7.98 on
AVEC 2013, and 6.00 and 7.75 on AVEC 2014 [16]. Supporting ablations report that vector-wise
fusion outperforms feature-wise fusion and either branch alone, that two split feature groups
are optimal, and that the eighteen-layer configuration outperforms deeper variants at lower
computational cost. The paper additionally provides visual analysis using an extended
Grad-CAM++ [17] and an assessment of robustness to degraded inputs.

Reference [16] was selected as the replication target for three reasons: the reported result
is competitive with the state of the art on both standard corpora; the method's central claim
is architectural and therefore testable by reimplementation; and the authors publish source
code and trained weights, and the article carries a third-party reproducibility certification.
In principle, this is close to a best case for computational replication.

## 1.4 Evaluation practice in this literature

The contribution of a replication depends on what is checked, and the preceding sections
suggest four things that are not routinely checked in this field.

**Variance is not reported.** Strong evidence that one method outperforms another requires
repeated trials across the sources of variation in the training pipeline — initialisation,
data ordering, augmentation sampling, and hyperparameter selection. Bouthillier et al. [18]
model this process and show that these sources contribute materially to measured performance,
and that comparisons which ignore them frequently attribute to method what is properly
attributed to chance. Sculley et al. [19] make the related argument that competitive
benchmarking without variance estimates systematically rewards fortunate runs. In a
literature where successive methods are separated by a few tenths of a point of mean
absolute error, and where no paper reports a variance estimate, it is not currently possible
to establish whether the reported ordering of methods is real.

**Model selection uses the reported partition.** Musgrave et al. [20] examined deep metric
learning and found that a substantial proportion of published work performed model selection
and hyperparameter tuning with direct feedback from the test set, producing an apparent
progression that largely disappeared once a validation set was reinstated. The AVEC corpora
provide a development partition precisely to avoid this, but its use is not consistently
documented in the depression estimation literature.

**Trivial baselines are absent.** DeMasi et al. [21] demonstrate, in a medical machine
learning context, that comparisons which omit simple baselines produce false optimism, since
a model may be outperformed by a constant predictor while still appearing to perform well in
absolute terms. Given that the Beck Depression Inventory-II distribution on the AVEC corpora
is concentrated in the minimal and mild bands, a constant predictor is a demanding baseline
on these data, yet none of the work surveyed above reports one.

**Participants are not disjoint across partitions.** Lopez-Otero et al. [22] observed that
certain speakers appear in more than one partition of the AVEC 2014 corpus, and that results
computed on the official partitions are biased as a consequence; subsequent work in the
speech modality has responded by adopting leave-one-speaker-out protocols. This observation
appears not to have propagated to the facial video literature, where the official partitions
remain in standard use.

To these four may be added a fifth concern specific to the reproducibility of published
artefacts rather than of results, which Section 4 addresses directly on the basis of evidence
obtained during this work.

---

## 1.5 Aims and objectives

*[Awaiting your wording. To be stated as agreed with your supervisor at the outset, and
not revised in light of the outcome.]*

---

# References

### Verified — details confirmed

- **[16]** Y. Pan, Y. Shang, T. Liu, Z. Shao, G. Guo, H. Ding, and Q. Hu, "Spatial–Temporal
  Attention Network for Depression Recognition from facial videos," *Expert Systems With
  Applications*, vol. 237, art. 121410, 2024.
- **[17]** A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, "Grad-CAM++:
  Generalized gradient-based visual explanations for deep convolutional networks," in *Proc.
  IEEE Winter Conf. Applications of Computer Vision*, 2018, pp. 839–847.
- **[18]** X. Bouthillier et al., "Accounting for variance in machine learning benchmarks,"
  in *Proc. Machine Learning and Systems (MLSys)*, 2021.
- **[20]** K. Musgrave, S. Belongie, and S.-N. Lim, "A metric learning reality check," in
  *Proc. European Conf. Computer Vision*, 2020, pp. 681–699.
- **[21]** O. DeMasi, K. Kording, and B. Recht, "Meaningless comparisons lead to false
  optimism in medical machine learning," *PLOS ONE*, vol. 12, no. 9, e0184604, 2017.
- **[13]** K. Xu et al., "Show, attend and tell: Neural image caption generation with visual
  attention," in *Proc. 32nd Int. Conf. Machine Learning*, 2015, pp. 2048–2057.
- **[14]** S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention
  module," in *Computer Vision — ECCV 2018*, Springer, pp. 3–19.

### Taken from the bibliography of [16] — verify page numbers and volumes before submission

- **[2]** M. Fava and K. Kendler, "Major depressive disorder," *Neuron*, vol. 28, no. 2,
  pp. 335–341, 2000.
- **[5]** Y. Zhu, Y. Shang, Z. Shao, and G. Guo, "Automated depression diagnosis based on deep
  networks to encode facial appearance and dynamics," *IEEE Trans. Affective Computing*,
  vol. 9, no. 4, pp. 578–584, 2018.
- **[6]** X. Zhou, K. Jin, Y. Shang, and G. Guo, "Visually interpretable representation
  learning for depression recognition from facial images," *IEEE Trans. Affective Computing*,
  vol. 11, no. 3, pp. 542–552, 2020.
- **[8]** W. C. de Melo, E. Granger, and A. Hadid, "Combining global and local convolutional
  3D networks for detecting depression from facial expressions," in *Proc. IEEE Int. Conf.
  Automatic Face and Gesture Recognition*, 2019, pp. 1–8.
- **[9]** W. C. de Melo, E. Granger, and A. Hadid, "A deep multiscale spatiotemporal network
  for assessing depression from facial dynamics," *IEEE Trans. Affective Computing*, 2020.
- **[11]** W. C. de Melo, E. Granger, and A. Hadid, "Depression detection based on deep
  distribution learning," in *Proc. IEEE Int. Conf. Image Processing*, 2019, pp. 4544–4548.
- **[12]** X. Zhou, Z. Wei, M. Xu, S. Qu, and G. Guo, "Facial depression recognition by deep
  joint label distribution and metric learning," *IEEE Trans. Affective Computing*, 2020.

### INCOMPLETE — do not submit until you have located these yourself

- **[1]** Ackerman et al., 2018 — cited by [16] for depression prevalence across age groups.
  Full details not recovered; consider replacing with a WHO or epidemiological source you
  can verify.
- **[3]** M. Valstar, B. Schuller, K. Smith, F. Eyben, et al., AVEC 2013 challenge paper.
  Full citation required. Also the source for the BDI-II severity bands — you may prefer to
  cite Beck's original instrument directly.
- **[4]** M. Valstar, B. Schuller, K. Smith, T. Almaev, et al., AVEC 2014 challenge paper.
  Full citation required.
- **[7]** Jazaery and Guo, 2018. Full citation required.
- **[10]** W. Carneiro de Melo, E. Granger, and M. Bordallo Lopez, 2021 — the MDN paper.
  Full citation required.
- **[15]** X. Wang et al., "Non-local neural networks," 2018. Full citation required.
- **[19]** D. Sculley, J. Snoek, A. Wiltschko, and A. Rahimi, "Winner's curse? On pace,
  progress, and empirical rigor," 2018. Venue needs confirming — appeared as an ICLR
  workshop contribution.
- **[22]** Lopez-Otero et al. — **the most important one to verify.** The observation about
  speaker overlap in AVEC 2014 is attributed to a 2015 Lopez-Otero paper, cited in
  "Analysis of gender and identity issues in depression detection on de-identified speech"
  (*Computer Speech & Language*). Locate the 2015 original and cite that, not the paper
  citing it. Your Section 1.4 claim rests on this reference.

*Numbering is provisional and will renumber once the incomplete entries are resolved.*
