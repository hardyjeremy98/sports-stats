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

type View = "assessment" | "roadmap" | "UI";

const capabilityRows = [
  ["Short-term tracking", "Implemented", "BoT-SORT and IoU; learned MOTR is a stub", "Good"],
  ["Team classification", "Implemented", "Kit-color and SigLIP clustering", "Good"],
  ["Cross-tracklet association", "Implemented, weak evidence", "Team/time/speed constraints plus mean torso color", "Good shell, weak core"],
  ["High-fidelity face anchors", "Implemented", "Largest boxes, face-score gate, weighted embeddings, optional SR", "Useful prototype"],
  ["Body re-ID", "Not implemented", "A documented replacement seam exists in the associator", "Medium effort"],
  ["Gait / motion identity", "Not implemented", "No temporal embedding or modality artifact yet", "High effort"],
  ["Multimodal fusion", "Not implemented", "Would currently need one composite resolver/associator", "Medium–high effort"],
  ["Roster identity scoring", "Not implemented", "GT jersey/team exist but identity labels are not scored", "Blocking gap"],
];

const uiRows = [
  ["Identity failure browser", "Extend existing eval switch list; highlight both tracklet and entity; add timeline markers", "P0"],
  ["Anchor-frame inspector", "Click crop to seek, zoom, inspect full-frame context, raw/upscaled pair and quality fields", "P0"],
  ["Eval-aware run diff", "Same-video guard, fixed/new switches, identity deltas, synchronized playhead", "P0"],
  ["Association inspector", "Entity-to-tracklet timeline, candidate merge scores, constraints accepted/rejected", "P1"],
  ["Identity QA", "Same/different person, merge/split, roster assignment; export crop pairs and provenance", "P1"],
  ["Batch benchmark view", "Sequences × configs × seeds with aggregate IDF1, swaps, purity and coverage", "P2"],
];

const roadmap = [
  {
    title: "1 · Make ID measurable",
    body: "Add a third evaluation level for semantic roster identity, plus team accuracy, cluster purity/coverage and anchor coverage. Until this exists, face vs none cannot move eval.json.",
    outcome: "A new ID strategy can win or lose objectively.",
  },
  {
    title: "2 · Establish the real baseline",
    body: "Upgrade eval-pipelines to run GT evaluation over many sequences and aggregate IDF1, ID switches, fragmentations and semantic-ID metrics. Preserve config, model version and seed.",
    outcome: "Repeatable regression science rather than single-run inspection.",
  },
  {
    title: "3 · Replace within-team colour affinity",
    body: "Keep the upstream team separation and existing team, overlap, gap and speed constraints, but replace mean Lab torso colour as the within-team player affinity with a learned body re-ID embedding and quality-weighted crop aggregation.",
    outcome: "The fastest likely gain using the architecture already present.",
  },
  {
    title: "4 · Generalize anchor evidence",
    body: "Extract a reusable anchor artifact with size, blur, pose, occlusion and modality quality. Persist embeddings/evidence so face, body, attributes and later gait do not repeatedly decode video.",
    outcome: "A foundation for quality-guided multimodal fusion.",
  },
  {
    title: "5 · Close the inference loop",
    body: "Let identity evidence influence merge/split decisions through iterative global inference or a joint optimizer. Today face labels can reveal a bad association but cannot repair it.",
    outcome: "Backfill identity becomes part of tracking, not downstream decoration.",
  },
];

function StatusStrip() {
  return (
    <Grid columns={4} gap={12}>
      <Stat value="Strong" label="Pipeline modularity" tone="success" />
      <Stat value="Good" label="Tracking/association eval" tone="success" />
      <Stat value="Early" label="ID evidence stack" tone="warning" />
      <Stat value="Missing" label="Roster-ID benchmark" tone="danger" />
    </Grid>
  );
}

function Assessment() {
  const action = useCanvasAction();
  return (
    <Stack gap={18}>
      <Callout tone="info" title="Bottom line">
        PitchLab is a well-shaped v1 research platform for tracker and cross-tracklet association
        experiments. It is not yet able to prove that face, body, gait or multimodal identity
        improves true player identification, because the semantic identity output is downstream of
        association and absent from evaluation.
      </Callout>

      <StatusStrip />

      <Grid columns="1.15fr 0.85fr" gap={16}>
        <Stack gap={10}>
          <H2>What is genuinely working</H2>
          <Text>
            The fixed stage registry, YAML-selected implementations, run-directory artifacts and
            Lab diff make tracker and associator swaps low-friction. Ground-truth evaluation already
            distinguishes raw tracklet identity from post-association entity identity.
          </Text>
          <Table
            headers={["Capability", "State", "Current implementation", "Readiness"]}
            rows={capabilityRows}
            rowTone={["success", "success", "warning", "success", "danger", "danger", "danger", "danger"]}
            striped
          />
        </Stack>

        <Stack gap={12}>
          <Card size="lg">
            <CardHeader trailing={<Pill size="sm">Highest leverage</Pill>}>
              The associator seam
            </CardHeader>
            <CardBody>
              <Stack gap={9}>
                <Text>
                  The current global associator already applies team, temporal overlap, gap and
                  pixel-speed constraints. Its identity affinity is only mean torso color.
                </Text>
                <Text tone="secondary">
                  Replacing that feature with a quality-weighted body re-ID embedding is the
                  clearest first model experiment. The source explicitly documents this seam.
                </Text>
                <Button
                  variant="secondary"
                  onClick={() =>
                    action({
                      type: "openFile",
                      path: "packages/pitchlab_core/src/pitchlab_core/stages/associate/global_embed.py",
                      selection: { startLineNumber: 1, startColumn: 1, endLineNumber: 16, endColumn: 1 },
                    })
                  }
                >
                  Open associator
                </Button>
              </Stack>
            </CardBody>
          </Card>

          <Card>
            <CardHeader>Face-anchor implementation</CardHeader>
            <CardBody>
              <Stack gap={8}>
                <Text>
                  Largest player boxes select candidate frames; face detections are score-gated;
                  embeddings are quality-weighted; evidence crops are saved; optional RealESRGAN is
                  available.
                </Text>
                <Text tone="secondary">
                  This is a useful prototype of your high-fidelity-frame idea, but “largest box” is
                  only a partial quality model and face clustering currently labels entities after
                  association.
                </Text>
                <Button
                  variant="ghost"
                  onClick={() =>
                    action({
                      type: "openFile",
                      path: "packages/pitchlab_core/src/pitchlab_core/stages/identity/face.py",
                      selection: { startLineNumber: 69, startColumn: 1, endLineNumber: 152, endColumn: 1 },
                    })
                  }
                >
                  Open face resolver
                </Button>
              </Stack>
            </CardBody>
          </Card>

          <Callout tone="warning" title="Architectural mismatch">
            The runner executes associate before identity. Face/body evidence can expose duplicate or
            incorrect entities, but cannot currently split or merge them. Your backfill concept
            ultimately needs iterative or joint global inference.
          </Callout>
        </Stack>
      </Grid>

      <Divider />
      <H2>What the current numbers do—and do not—mean</H2>
      <Grid columns={3} gap={14}>
        <Card variant="borderless">
          <CardHeader>Tracklet level</CardHeader>
          <CardBody>
            <Text>Scores raw tracker IDs against GT tracks using IDF1, MOTA, switches and fragmentation.</Text>
          </CardBody>
        </Card>
        <Card variant="borderless">
          <CardHeader>Entity level</CardHeader>
          <CardBody>
            <Text>Scores post-association player_id groupings. This is the useful metric for a new associator.</Text>
          </CardBody>
        </Card>
        <Card variant="borderless">
          <CardHeader>Semantic identity</CardHeader>
          <CardBody>
            <Text>
              Not scored. Face label, jersey/roster identity, team correctness and modality evidence do
              not affect eval.json.
            </Text>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

function Roadmap() {
  return (
    <Stack gap={16}>
      <Callout tone="warning" title="Order matters">
        Do not add gait or a complex fusion model first. The immediate bottleneck is experimental
        validity: semantic ID cannot yet be scored, and batch evaluation summarizes counts rather
        than GT accuracy.
      </Callout>
      <Grid columns="minmax(0, 1fr) minmax(0, 1fr)" gap={14}>
        {roadmap.map((item, index) => (
          <div key={item.title}>
            <Card size={index === 0 ? "lg" : "base"}>
              <CardHeader trailing={index < 2 ? <Pill size="sm">P0</Pill> : undefined}>
                {item.title}
              </CardHeader>
              <CardBody>
                <Stack gap={9}>
                  <Text>{item.body}</Text>
                  <Text tone="secondary" size="small">
                    Outcome: {item.outcome}
                  </Text>
                </Stack>
              </CardBody>
            </Card>
          </div>
        ))}
      </Grid>
      <H2>Recommended experiment ladder</H2>
      <Table
        headers={["Experiment", "Compare", "Primary metric", "Decision"]}
        rows={[
          ["Association null baseline", "per-tracklet vs global-color", "Entity IDF1, ID switches", "Does current association help at all?"],
          ["Body embedding", "Within-team colour vs body re-ID", "Entity IDF1 gain by condition", "Does learned appearance distinguish teammates better?"],
          ["Anchor weighting", "uniform vs quality-weighted", "Retrieval mAP + entity IDF1", "Does high-fidelity selection help?"],
          ["Face contribution", "body only vs body+face", "Roster-ID purity at fixed coverage", "Is face worth cost/licensing?"],
          ["Constraints", "appearance only vs constraint stack", "Silent-swap rate", "Which constraints carry the system?"],
          ["Gait contribution", "best stack vs +gait", "Marginal ID gain by crop quality", "Keep or kill gait."],
        ]}
        striped
      />
    </Stack>
  );
}

function UIReview() {
  const action = useCanvasAction();
  return (
    <Stack gap={16}>
      <Grid columns="1fr 1fr" gap={16}>
        <Stack gap={9}>
          <H2>Current Lab strengths</H2>
          <Text>
            It is already a good forensic viewer for GT-labelled clips: scrub source video, toggle
            predictions and GT, inspect tracklets/entities, view IDF1 before and after association,
            click ID-switch events, and inspect saved face evidence.
          </Text>
        </Stack>
        <Callout tone="warning" title="Current UX ceiling">
          It is not yet a re-ID workbench. Evidence crops are passive 48px thumbnails, entity-level
          failures are weakly highlighted, run diff has no synchronized video/eval comparison, and
          QA corrects event attribution rather than identity merges, splits or roster labels.
        </Callout>
      </Grid>

      <H2>UI improvements in priority order</H2>
      <Table
        headers={["Feature", "Research workflow enabled", "Priority"]}
        rows={uiRows}
        rowTone={["danger", "danger", "danger", "warning", "warning", "neutral"]}
        striped
      />

      <Grid columns="1.1fr 0.9fr" gap={16}>
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm">Suggested core screen</Pill>}>
            Identity investigation workspace
          </CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Grid columns="150px 1fr 220px" gap={10}>
                <Stack gap={6}>
                  <H3>Failure list</H3>
                  <Text size="small" tone="secondary">
                    Switches, low confidence, cluster collisions, unknown identity.
                  </Text>
                </Stack>
                <Stack gap={6}>
                  <H3>Video + synchronized timeline</H3>
                  <Text size="small" tone="secondary">
                    GT and predicted identity, anchor markers, previous/next tracklet context.
                  </Text>
                </Stack>
                <Stack gap={6}>
                  <H3>Evidence + decision</H3>
                  <Text size="small" tone="secondary">
                    Face/body/attribute scores; same/different, merge/split, assign roster player.
                  </Text>
                </Stack>
              </Grid>
              <Divider />
              <Text tone="secondary">
                This turns the existing run viewer, evidence crops and QA primitives into one
                closed-loop tool whose corrections become re-ID training pairs.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Small fixes with immediate value</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Make every evidence crop seek to its source frame and open at useful size.</Text>
              <Text>Highlight entity-level switch IDs, not only raw tracklets.</Text>
              <Text>Require same video for diff and make both timelines seekable.</Text>
              <Text>Expose the existing re-evaluate endpoint in the run viewer.</Text>
              <Text>Add GT/config/metric filters and sorting to the run dashboard.</Text>
              <Button
                variant="secondary"
                onClick={() =>
                  action({
                    type: "openFile",
                    path: "web/src/pages/LabRunViewer.tsx",
                    selection: { startLineNumber: 332, startColumn: 1, endLineNumber: 405, endColumn: 1 },
                  })
                }
              >
                Open current Players UI
              </Button>
            </Stack>
          </CardBody>
        </Card>
      </Grid>
    </Stack>
  );
}

export default function PlayerIDExperimentationReview() {
  const theme = useHostTheme();
  const [view, setView] = useCanvasState<View>("review-view", "assessment");

  return (
    <div style={{ minHeight: "100%", background: theme.bg.editor, color: theme.text.primary, padding: 24 }}>
      <Stack gap={18}>
        <Stack gap={7}>
          <Text size="small" tone="tertiary">
            PITCHLAB · REPOSITORY REVIEW
          </Text>
          <H1>Player ID experimentation readiness</H1>
          <Text tone="secondary">
            Assessment of current implementation, measurement gaps, and the Lab changes that would
            make face/body/gait and global identity strategies fast to test.
          </Text>
        </Stack>

        <Row gap={8} wrap>
          <Pill active={view === "assessment"} onClick={() => setView("assessment")}>
            Current state
          </Pill>
          <Pill active={view === "roadmap"} onClick={() => setView("roadmap")}>
            Recommended sequence
          </Pill>
          <Pill active={view === "UI"} onClick={() => setView("UI")}>
            Lab UI
          </Pill>
        </Row>

        {view === "assessment" && <Assessment />}
        {view === "roadmap" && <Roadmap />}
        {view === "UI" && <UIReview />}

        <Divider />
        <Text size="small" tone="quaternary">
          Evidence reviewed: pipeline runner and interfaces, global-color association, face identity,
          MOT evaluation, batch experiments, artifacts/APIs, Lab run viewer, diff and QA flows.
        </Text>
      </Stack>
    </div>
  );
}
