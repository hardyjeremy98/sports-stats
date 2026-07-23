import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Link,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type View = "thesis" | "current" | "roadmap" | "candidates" | "experiments";

const detectorRows = [
  [
    "Hosted football-players-detection-3zvbc/11",
    "Current shipping prototype",
    "Broadcast-derived soccer data; exact checkpoint lineage and hosted preprocessing are not pinned",
    "Useful baseline, not validated for phone footage",
  ],
  [
    "Local YOLOv8x football-player-detection.pt",
    "Local benchmark only",
    "1280px; upstream dataset/checkpoint version appears older than hosted model",
    "AGPL runtime/weights: do not ship without commercial licensing",
  ],
  [
    "RF-DETR Medium / Large",
    "Recommended permissive challenger",
    "Apache-2.0 core/weights; benchmark on labelled sports, then adapt on owned phone footage",
    "Best first shipping-path experiment",
  ],
  [
    "D-FINE / RT-DETRv2",
    "Recommended localization challengers",
    "Permissive implementations; review each checkpoint's pretraining lineage",
    "Benchmark on identical data and resolution",
  ],
  [
    "YOLOX-X",
    "Literature reproduction baseline",
    "Apache-2.0 and well represented in sports MOT papers",
    "Good reference for Deep-EIoU-style experiments",
  ],
];

const trackerRows = [
  [
    "Current BoT-SORT-style",
    "Kalman + IoU/score association + CMC; no learned appearance in installed implementation",
    "Strong licensed baseline",
    "Not SOTA",
  ],
  [
    "Full BoT-SORT + ReID",
    "Adds online appearance matching while retaining camera compensation",
    "Mature next baseline",
    "High priority",
  ],
  [
    "TDLP",
    "Learns track-to-detection link probabilities from bbox history with optional appearance and pose",
    "MIT code, official SportsMOT weights and broad benchmark coverage",
    "Highest-priority learned tracker",
  ],
  [
    "OC-SORT",
    "Observation-centric recovery from nonlinear motion and occlusion",
    "Lightweight ablation",
    "Medium priority",
  ],
  [
    "Deep-EIoU",
    "ExpansionIoU + iterative matching + OSNet appearance; sports-specific evidence",
    "Strong research upper baseline; official repo lacks clear license",
    "Reproduce or reimplement carefully",
  ],
  [
    "CAMELTrack",
    "Learned modular association using box, appearance and optional keypoints",
    "Promising licensed learned-association candidate",
    "After mature baselines",
  ],
  [
    "Selective SAM2 correction",
    "Invoke masks only for ambiguous assignment windows",
    "Potential purity tool without full mask-tracker cost",
    "Targeted research experiment",
  ],
  [
    "SAMIDARE / HyperSSM / NOOUGAT",
    "Recent mask, state-space, hypergraph or offline graph systems",
    "SOTA references with immature or unavailable implementations",
    "Watch, do not lead with",
  ],
];

const metricRows = [
  ["Detector recall", "Player recall by box-height bin; consecutive miss-burst length", "Miss bursts fragment tracks even when frame-average recall looks good"],
  ["Detector localization", "AP75, IoU distribution, bottom-centre error, temporal box jitter", "Loose/jittery boxes damage IoU matching and contaminate body crops"],
  ["Raw identity", "Tracklet IDF1, ID precision/recall, ID switches, fragmentations", "Primary score for online tracklet quality"],
  ["Tracklet purity", "GT identities per predicted tracklet; mixed-track duration", "Directly catches contaminated tracklets that global merging cannot repair"],
  ["Fragmentation", "Predicted tracklets per GT player; track-length distribution", "Separates repairable breaks from silent swaps"],
  ["Detection/association balance", "HOTA with DetA, AssA and localization component", "Not implemented yet; needed for comparison with sports literature"],
  ["Identity evidence yield", "Quality-approved body/face crops per GT player", "A detector can track adequately while starving downstream identity"],
  ["Post-tracklet association", "Entity IDF1 gain and merge precision", "Measures the body re-ID work currently in progress, not raw tracklets"],
];

const candidateRows = [
  ["Deep-EIoU + GTA", "SportsMOT 81.04 HOTA; SoccerNet up to 83.11 HOTA", "Strong split/connect reference; broadcast domain; Deep-EIoU licensing unclear"],
  ["GTATrack 2025", "SoccerTrack winner, 0.60 HOTA", "YOLO11x + Deep-EIoU + GTA on fixed fisheye amateur footage; different protocol"],
  ["TDLP", "SportsMOT 81.9 HOTA; +1.5 HOTA and +3.1 IDF1 over CAMELTrack", "Official MIT implementation and weights; bbox-only and appearance variants make body-ReID contribution directly measurable"],
  ["CAMELTrack", "SportsMOT 80.3 HOTA", "Apache-2.0 modular learned association; keypoints reportedly hurt SportsMOT"],
  ["HyperSSM", "SportsMOT 78.5 HOTA", "CVPR 2026 nonlinear collaborative motion; no usable release found"],
  ["SAM2MOT", "DanceTrack 75.8 HOTA", "Apache-2.0 but heavy; mask drift and dynamic tuning risks"],
  ["SAMIDARE", "SportsMOT val 86.2 HOTA; test reproduction 77.3", "2026 mask regeneration/reactivation; protocol-sensitive and immature repository"],
  ["Selective SAM2/3 + tracker + GTA", "Claimed SportsMOT test up to 87.2 HOTA", "Very new bundled system; selective ambiguity correction is more useful than headline score"],
  ["NOOUGAT offline", "SportsMOT 85.6 HOTA", "Compelling graph association reference; no official implementation found"],
];

const datasetRows = [
  ["SportsMOT", "Basketball, football and volleyball broadcast clips", "Fast algorithm selection, cross-sport robustness and literature comparison", "Easy / abundant"],
  ["SoccerNet Tracking", "Professional broadcast soccer", "Soccer-specific tracking and GT identity continuity", "Easy / relevant"],
  ["SoccerTrack", "Fixed panoramic/fisheye amateur soccer", "Scale, distortion and amateur-player bridge", "Intermediate"],
  ["AFMOT or comparable handheld sports", "Handheld American-football-style footage", "Camera motion, blur, collisions and zoom transfer", "Hard"],
  ["Owned phone soccer", "Ground-level target footage", "Final adaptation and product acceptance", "Hardest / scarce"],
];

const phaseRows = [
  {
    phase: "0",
    title: "Establish a truthful ceiling",
    gate: "Can we separate detector failure from tracker failure?",
    actions: [
      "Add detection-only GT scoring and HOTA/DetA/AssA.",
      "Run GT boxes through each tracker to estimate the detector-independent ceiling.",
      "Record model SHA, data revision, package version, Git revision and evaluation-set hash.",
      "Add predicted-tracklet purity and mixed-track duration.",
    ],
    exit: "Every missed track or switch can be attributed to detection, local association, or post-tracklet association.",
  },
  {
    phase: "1",
    title: "Harden the existing baseline",
    gate: "What can the current stack achieve before replacing models?",
    actions: [
      "Sweep stride, detector confidence, lost buffer, activation threshold, minimum length and CMC.",
      "Remove the silent zero-argument tracker-constructor fallback.",
      "Preserve person class through tracking instead of assigning every detection class 0.",
      "Distinguish observed, predicted and interpolated boxes in artifacts; do not imply current boxes are smoothed.",
    ],
    exit: "A reproducible BoT-SORT-style baseline on held-out SportsMOT and SoccerNet sequences.",
  },
  {
    phase: "2",
    title: "Build the public-sports detector ladder",
    gate: "Which detector produces the best downstream tracklets across labelled sports domains?",
    actions: [
      "Start with SportsMOT, SoccerNet and SoccerTrack using source-separated train/validation/test splits.",
      "Compare hosted v11 where applicable, RF-DETR M/L, D-FINE, RT-DETRv2 and permissive YOLOX on identical data.",
      "Measure small-player recall, miss bursts, duplicates, localization jitter and quality-crop survival.",
      "Test stride 1 vs 2, model-native resolution and selective tiling before generic TTA.",
    ],
    exit: "Two licensed detector candidates selected by raw tracklet IDF1, purity and cross-sport consistency.",
  },
  {
    phase: "3",
    title: "Add appearance to online tracking",
    gate: "Does body appearance prevent raw tracklet switches among teammates?",
    actions: [
      "Finish the current body re-ID work offline and validate within-team discrimination first.",
      "Freeze the offline associator while evaluating online use of the same encoder.",
      "Compare current tracker, full BoT-SORT+ReID, TDLP bbox-only, TDLP+body-ReID, OC-SORT and a Deep-EIoU reference.",
      "Quality-gate appearance so low-resolution or occluded crops cannot force a match.",
    ],
    exit: "Raw tracklet ID switches fall without unacceptable wrong continuations or compute cost.",
  },
  {
    phase: "4",
    title: "Make purity the control policy",
    gate: "Can one policy detect ambiguity before creating mixed tracklets across sports?",
    actions: [
      "Log assignment margins and competing candidate scores.",
      "Prefer terminating a tracklet over forcing a near-tied assignment.",
      "Split tracklets where appearance or motion changes abruptly.",
      "Try CAMELTrack and selective SAM2 only on measured hard windows; interpolate only settled short gaps.",
    ],
    exit: "Mixed-track duration falls on multiple sports without dataset-specific threshold overfitting.",
  },
  {
    phase: "5",
    title: "Transfer the finalists to phone footage",
    gate: "Which gains survive the move from labelled benchmarks to scarce handheld target data?",
    actions: [
      "Take only the best two detector/tracker stacks into AFMOT-style handheld and owned phone evaluation.",
      "Fine-tune on a small match-separated phone set; use temporally consistent pseudo-labels to expand training data.",
      "Re-tune camera compensation, rolling-shutter/blur handling and field registration for phone motion.",
      "Adopt selective masks or newer SOTA systems only when they improve the remaining phone-specific failure windows.",
    ],
    exit: "The chosen stack survives held-out phone matches, licensing review and cost-per-match limits.",
  },
];

function PipelineBand() {
  const theme = useHostTheme();
  const stages = [
    ["Pixels", "sports video"],
    ["Detect", "boxes + scores"],
    ["Track", "raw tracklets"],
    ["Associate", "player entities"],
    ["Identify", "roster labels"],
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(5, minmax(0, 1fr))", gap: 6 }}>
      {stages.map(([name, detail], i) => (
        <div
          key={name}
          style={{
            padding: "12px 10px",
            border: `1px solid ${i === 2 ? theme.accent.primary : theme.stroke.secondary}`,
            background: i === 2 ? theme.fill.secondary : theme.fill.quaternary,
            borderRadius: 6,
          }}
        >
          <Text as="span" weight="semibold">{name}</Text>
          <Text size="small" tone="tertiary" style={{ marginTop: 3 }}>{detail}</Text>
        </div>
      ))}
    </div>
  );
}

function ThesisView() {
  return (
    <Stack gap={18}>
      <Callout tone="warning" title="Decision">
        The current tracklet system is a credible baseline, not SOTA. The architecture is correct,
        and labelled broadcast/fixed-camera sports footage is the right first proving ground. The
        installed BoT-SORT-style tracker still uses IoU, score, Kalman prediction and camera
        compensation without learned appearance; phone transfer comes after benchmark selection.
      </Callout>

      <PipelineBand />

      <Grid columns={4} gap={12}>
        <Stat value="Baseline" label="Current raw tracker" tone="warning" />
        <Stat value="Public first" label="Benchmark sequence" tone="success" />
        <Stat value="After track" label="Body re-ID wiring now" />
        <Stat value="Purity first" label="Roadmap objective" tone="success" />
      </Grid>

      <Grid columns="1.15fr 0.85fr" gap={16}>
        <Stack gap={10}>
          <H2>The tracklet contract</H2>
          <Text>
            A tracklet is a short sequence of boxes carrying one presumed physical identity. It is
            consumed by team classification, body/face evidence, global association, minimap fusion
            and event attribution. The target is not the longest possible track—it is the longest
            track that remains identity-pure.
          </Text>
          <Callout tone="info" title="The key asymmetry">
            Fragmentation is repairable by offline association. A tracklet that silently changes
            from player A to player B is contaminated and must first be split. Therefore uncertain
            online matches should terminate rather than force continuity.
          </Callout>
        </Stack>

        <Card size="lg">
          <CardHeader>Body re-ID boundary</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                The in-progress body embedding currently compares completed tracklets in the
                offline associate stage. It can improve entity IDF1, but it cannot change raw
                tracklet IDF1 or repair an internal switch.
              </Text>
              <Text tone="secondary">
                Once offline within-team retrieval is validated, the same encoder can become a
                quality-gated cue inside online tracking. That is a separate experiment with a
                separate success metric.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Recommended destination</H2>
      <Text>
        Domain-trained detector → camera-aware local association using motion plus quality-gated body
        appearance → short high-purity tracklets → ambiguity-driven splitting → conservative offline
        global association. Select the architecture on broad labelled sports footage, bridge through
        amateur/fisheye and handheld sets, then spend scarce phone labels only on the finalists.
      </Text>

      <H2>Difficulty ladder</H2>
      <Table
        headers={["Dataset tier", "Footage", "Purpose", "Difficulty"]}
        rows={datasetRows}
        striped
      />
    </Stack>
  );
}

function CurrentView() {
  const action = useCanvasAction();
  return (
    <Stack gap={18}>
      <H2>Current raw-tracklet implementation</H2>
      <Grid columns="1fr 1fr" gap={14}>
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm">Detect</Pill>}>Bounding-box evidence</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                Production uses hosted `football-players-detection-3zvbc/11` at confidence 0.3,
                normally every second source frame. The model is soccer-specific but derived from
                professional broadcast imagery and lacks immutable provenance in run metadata.
              </Text>
              <Text tone="secondary">
                The local YOLOv8x path runs at 1280px but is AGPL and appears tied to a different
                dataset/checkpoint revision. It is a benchmark reference, not a shipping path.
              </Text>
              <Button
                variant="secondary"
                onClick={() => action({ type: "openFile", path: "packages/matchlab_core/src/matchlab_core/stages/detect/roboflow.py" })}
              >
                Open detector
              </Button>
            </Stack>
          </CardBody>
        </Card>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm">Track</Pill>}>Online association</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>
                The wrapper uses Roboflow `trackers` BoT-SORT-style tracking: Kalman prediction,
                camera-motion compensation, high/low-confidence IoU matching and a lost-track buffer.
              </Text>
              <Text tone="secondary">
                The installed implementation returns matched detection boxes, not Kalman-smoothed or
                gap-interpolated boxes. It has no learned appearance term. All person classes are
                submitted as class 0, then reconstructed by nearest centre.
              </Text>
              <Button
                variant="secondary"
                onClick={() => action({ type: "openFile", path: "packages/matchlab_core/src/matchlab_core/stages/track/botsort.py" })}
              >
                Open tracker wrapper
              </Button>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Detector and checkpoint audit</H2>
      <Table
        headers={["Candidate", "Role", "Evidence / uncertainty", "Verdict"]}
        rows={detectorRows}
        rowTone={["warning", "danger", "success", "info", "neutral"]}
        striped
      />

      <H2>Hidden risks worth fixing before model replacement</H2>
      <Table
        headers={["Risk", "Effect on tracklets", "Action"]}
        rows={[
          ["Every person detection enters tracker as class 0", "Player/GK/referee can be associated across role boundaries", "Preserve class or make class policy explicit"],
          ["Nearest-centre class recovery", "Overlapping players can inherit the wrong role", "Carry source detection index through assignment"],
          ["Silent zero-argument constructor fallback", "Tracker parameters can disappear after package API drift", "Fail loudly and version-pin"],
          ["Detector threshold 0.3 before tracker", "BoT-SORT cannot recover lower-score boxes", "Jointly tune detector and low-score association"],
          ["Minimum five observations", "Short valid tracks become misses", "Expose pre-filter and post-filter evaluation"],
          ["Stride 2", "One processed miss spans two source frames", "Benchmark stride as a quality/cost variable"],
          ["No observed/predicted provenance", "UI can imply continuity that artifacts do not contain", "Add frame-source flags before interpolation"],
        ]}
        striped
      />

      <Callout tone="warning" title="Do not select the next detector from public mAP">
        Use miss bursts, localization jitter, raw tracklet IDF1, mixed-track duration and
        identity-crop survival on each benchmark tier. Public sports data selects the finalists;
        held-out phone footage decides whether their gains transfer.
      </Callout>
    </Stack>
  );
}

function RoadmapView() {
  const theme = useHostTheme();
  return (
    <Stack gap={16}>
      <Callout tone="info" title="Ordering principle">
        Move from abundant/easier labels to scarce/harder labels. Each phase has an exit gate so
        phone annotation is spent only after weaker architectures have been eliminated.
      </Callout>
      {phaseRows.map((p) => (
        <div
          key={p.phase}
          style={{
            display: "grid",
            gridTemplateColumns: "52px minmax(0, 1fr) minmax(250px, 0.7fr)",
            gap: 14,
            padding: "14px 0",
            borderBottom: `1px solid ${theme.stroke.tertiary}`,
          }}
        >
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: 999,
              background: p.phase === "0" ? theme.accent.control : theme.fill.secondary,
              color: p.phase === "0" ? theme.text.onAccent : theme.text.primary,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 600,
            }}
          >
            {p.phase}
          </div>
          <Stack gap={7}>
            <H3>{p.title}</H3>
            <Text tone="secondary" size="small">{p.gate}</Text>
            <ul style={{ margin: 0, paddingLeft: 18, color: theme.text.secondary, fontSize: 13, lineHeight: 1.55 }}>
              {p.actions.map((a) => <li key={a}>{a}</li>)}
            </ul>
          </Stack>
          <div style={{ padding: 10, background: theme.fill.quaternary, borderRadius: 6 }}>
            <Text size="small" weight="semibold">Exit criterion</Text>
            <Text size="small" tone="secondary" style={{ marginTop: 5 }}>{p.exit}</Text>
          </div>
        </div>
      ))}
    </Stack>
  );
}

function CandidatesView() {
  return (
    <Stack gap={18}>
      <H2>Tracker candidates by implementation order</H2>
      <Table
        headers={["System", "Tracklet mechanism", "Readiness", "Recommendation"]}
        rows={trackerRows}
        rowTone={["warning", "success", "neutral", "info", "info", "warning", "danger"]}
        striped
      />

      <H2>Recent headline results—with protocol caveats</H2>
      <Table
        headers={["System", "Published result", "How to interpret it"]}
        rows={candidateRows}
        striped
      />
      <Text size="small" tone="tertiary">
        Sources span SportsMOT broadcast clips, SoccerNet broadcast footage, DanceTrack and fixed
        fisheye SoccerTrack. Their absolute numbers are not interchangeable and are upper-bound
        references for ground-level handheld footage.
      </Text>

      <Grid columns="1fr 1fr" gap={14}>
        <Card>
          <CardHeader>Implement soon</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>RF-DETR or D-FINE trained first on labelled sports, then adapted to phone data.</Text>
              <Text>Full BoT-SORT with quality-gated body ReID.</Text>
              <Text>TDLP bbox-only and TDLP+body-ReID as the first learned-association pair.</Text>
              <Text>OC-SORT and Deep-EIoU as controlled local-association baselines.</Text>
              <Text>GTA-style split-before-connect offline refinement.</Text>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Keep as research options</CardHeader>
          <CardBody>
            <Stack gap={6}>
              <Text>Selective SAM2 correction for near-tied crowded windows.</Text>
              <Text>CAMELTrack after appearance and bbox baselines exist.</Text>
              <Text>HyperSSM, NOOUGAT and SAMIDARE after code, weights and licenses mature.</Text>
              <Text>Pose only when player scale and keypoint confidence make it useful.</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Primary sources</H2>
      <Row gap={12} wrap>
        <Link href="https://arxiv.org/abs/2411.08216">GTA</Link>
        <Link href="https://arxiv.org/abs/2602.00484">GTATrack</Link>
        <Link href="https://arxiv.org/abs/2604.22162">SAMIDARE</Link>
        <Link href="https://arxiv.org/abs/2512.22105">TDLP</Link>
        <Link href="https://github.com/roboflow/rf-detr">RF-DETR</Link>
        <Link href="https://github.com/Peterande/D-FINE">D-FINE</Link>
        <Link href="https://github.com/Megvii-BaseDetection/YOLOX">YOLOX</Link>
        <Link href="https://github.com/noahcao/OC_SORT">OC-SORT</Link>
        <Link href="https://github.com/sjc042/gta-link">GTA-Link</Link>
      </Row>
    </Stack>
  );
}

function ExperimentsView() {
  return (
    <Stack gap={18}>
      <H2>Measurement stack required before “SOTA” claims</H2>
      <Table
        headers={["Layer", "Required measurement", "Why it changes the decision"]}
        rows={metricRows}
        striped
      />

      <H2>First eleven experiments</H2>
      <Table
        headers={["Order", "Controlled comparison", "Hold fixed", "Decision metric"]}
        rows={[
          ["1", "GT detections → current tracker", "Tracker params and evaluation frames", "Detector-independent tracklet ceiling"],
          ["2", "Current detector confidence 0.2 / 0.3 / 0.4 / 0.5", "Tracker and videos", "Miss bursts, FP, tracklet IDF1"],
          ["3", "Stride 1 / 2 / 4", "Detector, tracker and resolution", "IDF1, switches, runtime"],
          ["4", "CMC on / off", "Everything else", "Switches during camera motion"],
          ["5", "Lost buffer and minimum-length sweep", "Detector and CMC", "Fragmentation vs contamination"],
          ["6", "Hosted v11 / RF-DETR-M / D-FINE / RT-DETRv2", "Owned training split and tracker", "Raw IDF1, purity, crop yield"],
          ["7", "Current tracker / full BoT-SORT-ReID / OC-SORT", "Chosen detector and body encoder", "Raw switches and mixed-track duration"],
          ["8", "TDLP bbox-only / TDLP + body ReID", "Detector, TDLP weights and track management", "Marginal appearance gain in raw tracklets"],
          ["9", "Deep-EIoU reference", "Detector, ReID and test split", "AssA-equivalent metrics and compute"],
          ["10", "Forced match / terminate on low assignment margin", "All model scores", "Contamination vs fragmentation"],
          ["11", "Base tracker / selective SAM2 on ambiguous windows", "Detector and decision policy", "Hard-window purity and cost"],
        ]}
        striped
      />

      <Grid columns="1fr 1fr" gap={14}>
        <Callout tone="success" title="Body re-ID experiment now">
          Compare `global-color` and `global-reid` only at entity level: association gain and merge
          precision. Raw tracklet metrics must remain identical because both consume the same
          `tracklets.json`.
        </Callout>
        <Callout tone="warning" title="Body re-ID experiment later">
          Move the validated encoder into online detection-to-track matching. Then compare raw
          tracklet IDF1, switches and purity, while freezing offline association.
        </Callout>
      </Grid>

      <H2>Model provenance gate</H2>
      <Text>
        Every run intended for comparison should record model family and architecture revision,
        checkpoint SHA-256, package/runtime version, pretraining lineage and license, fine-tuning
        dataset manifest and split, training commit/seed, input transform, confidence/NMS/TTA,
        export precision, hardware, Git revision and evaluation-set hash.
      </Text>
    </Stack>
  );
}

export default function TrackletSOTARoadmap() {
  const theme = useHostTheme();
  const [view, setView] = useCanvasState<View>("tracklet-roadmap-view", "thesis");

  return (
    <div style={{ minHeight: "100%", background: theme.bg.editor, color: theme.text.primary, padding: 24 }}>
      <Stack gap={18}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1.5fr) minmax(240px, 0.5fr)",
            gap: 20,
            alignItems: "end",
            paddingBottom: 16,
            borderBottom: `1px solid ${theme.stroke.secondary}`,
          }}
        >
          <Stack gap={7}>
            <Text size="small" tone="tertiary">MATCHLAB · TRACKLET ENGINEERING REVIEW · JULY 2026</Text>
            <H1>From baseline tracks to SOTA-grade tracklets</H1>
            <Text tone="secondary">
              A broad-sports-first roadmap for detection, local association, camera motion,
              ambiguity and splitting—then transfer of the winning stack to casual phone footage.
            </Text>
          </Stack>
          <div style={{ padding: 12, background: theme.fill.tertiary, borderRadius: 6 }}>
            <Text size="small" tone="tertiary">Roadmap thesis</Text>
            <Text weight="semibold" style={{ marginTop: 4 }}>
              Improve evidence first. Preserve purity second. Add complexity only at measured failure windows.
            </Text>
          </div>
        </div>

        <Row gap={8} wrap>
          {([
            ["thesis", "Executive view"],
            ["current", "Current stack"],
            ["roadmap", "Gated roadmap"],
            ["candidates", "SOTA candidates"],
            ["experiments", "Experiments"],
          ] as [View, string][]).map(([id, label]) => (
            <div key={id}>
              <Pill active={view === id} onClick={() => setView(id)}>{label}</Pill>
            </div>
          ))}
        </Row>

        {view === "thesis" && <ThesisView />}
        {view === "current" && <CurrentView />}
        {view === "roadmap" && <RoadmapView />}
        {view === "candidates" && <CandidatesView />}
        {view === "experiments" && <ExperimentsView />}

        <Divider />
        <Text size="small" tone="quaternary">
          Repository evidence: current detector/tracker wrappers, installed Roboflow trackers 2026
          implementation, configs, GT evaluator and Lab artifacts. Research evidence: primary papers
          and repositories available through 16 July 2026. Public benchmark scores are not predictions
          of performance on MatchLab's target footage. The roadmap intentionally uses easier,
          abundant sports labels before scarce phone labels.
        </Text>
      </Stack>
    </div>
  );
}
