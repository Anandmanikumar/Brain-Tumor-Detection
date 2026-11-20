"use client";

import { useEffect, useRef, useState } from "react";

type LabelStat = {
  label: number;
  count: number;
  volumeCm3: number;
  percentage: number;
};

export default function MriViewer() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const niivueRef = useRef<any>(null);
  const [image, setImage] = useState<File | null>(null);
  const [mask, setMask] = useState<File | null>(null);
  const [opacity, setOpacity] = useState(0.45);
  const [showMask, setShowMask] = useState(true);
  const [loading, setLoading] = useState(false);
  // Segmentation analytics
  const [labelStats, setLabelStats] = useState<LabelStat[]>([]);
  const [totalVolumeCm3, setTotalVolumeCm3] = useState(0);

  const [visibleLabels, setVisibleLabels] = useState<
    Record<number, boolean>
  >({});
  

  

  const [message, setMessage] = useState(
    "Choose an MRI volume and its matching segmentation mask to open the viewer."
  );

  // Initialize and load the two local NIfTI volumes.
  useEffect(() => {
    let cancelled = false;
    let nv: any = null;
    let imageUrl: string | null = null;
    let maskUrl: string | null = null;

    async function loadViewer() {
      if (!canvasRef.current || !image || !mask) return;

      setLoading(true);
      setMessage("Initializing MRI viewer...");

      try {
        const { Niivue } = await import("@niivue/niivue");

        if (cancelled || !canvasRef.current) return;

        const viewer = new Niivue({
          backColor: [0.035, 0.055, 0.075, 1],
          show3Dcrosshair: true,
          isColorbar: false,
          crosshairColor: [0.2, 0.85, 1, 0.9],
        });

        nv = viewer;
        niivueRef.current = viewer;

        await viewer.attachToCanvas(canvasRef.current);

        if (cancelled) return;

        imageUrl = URL.createObjectURL(image);
        maskUrl = URL.createObjectURL(mask);

        
        await viewer.loadVolumes([
          {
            url: imageUrl,
            name: image.name,
          },
          {
            url: maskUrl,
            name: mask.name,
            opacity,
          },
        ]);

        // Apply a discrete color map to the segmentation volume.
        
        // Get the loaded segmentation volume.
        const segmentation = viewer.volumes[1];

        // Read dimensions and voxel spacing from NIfTI header.
        const dimensions = segmentation.hdr?.dims.slice(1, 4) ?? [];

        const spacing = segmentation.hdr?.pixDims.slice(1, 4) ?? [];

        // Calculate the physical volume of one voxel in mm³.
        const voxelVolumeMm3 =
          Number(spacing[0]) *
          Number(spacing[1]) *
          Number(spacing[2]);

        // Validate the metadata before calculating physical volumes.
        if (
          dimensions.length !== 3 ||
          spacing.length !== 3 ||
          !dimensions.every((v: number) => Number.isFinite(v) && v > 0) ||
          !Number.isFinite(voxelVolumeMm3) ||
          voxelVolumeMm3 <= 0
        ) {
          throw new Error("Invalid segmentation dimensions or voxel spacing.");
        }

        // Count voxels belonging to each non-background label.
        const counts = new Map<number, number>();

        const voxelData = segmentation.img as ArrayLike<number>;

        for (let i = 0; i < voxelData.length; i++) {
          const label = Number(voxelData[i]);

          if (label === 0) continue;

          counts.set(label, (counts.get(label) ?? 0) + 1);
        }

        // Calculate total number of segmented voxels.
        const totalVoxels = Array.from(counts.values()).reduce(
          (sum, count) => sum + count,
          0
        );

        


        // Generate statistics for every non-background label.
        const stats: LabelStat[] = Array.from(counts.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([label, count]) => ({
            label,
            count,
            volumeCm3: (count * voxelVolumeMm3) / 1000,
            percentage: totalVoxels > 0
              ? (count / totalVoxels) * 100
              : 0,
          }));

        setLabelStats(stats);

        setVisibleLabels(
          Object.fromEntries(
            stats.map((item) => [item.label, true])
          )
        );

        setTotalVolumeCm3(
          (totalVoxels * voxelVolumeMm3) / 1000
        );

        console.log("SEGMENTATION ANALYTICS", {
          dimensions,
          spacing,
          totalVoxels,
          totalVolumeCm3:
            (totalVoxels * voxelVolumeMm3) / 1000,
          stats,
        });
        

        segmentation.setColormapLabel({
          // Background: transparent
          // Label 1: Coral
          // Label 2: Amber
          // Label 3: Magenta
          R: [0, 255, 255, 220],
          G: [0, 80, 180, 50],
          B: [0, 40, 0, 220],
          A: [0, 255, 255, 255],
          I: [0, 1, 2, 3],
          labels: [
            "Background",
            "Label 1",
            "Label 2",
            "Label 3",
          ],
        });


        if (cancelled) return;

        // Display axial, coronal and sagittal views together.
        viewer.setSliceType(viewer.sliceTypeMultiplanar);
        viewer.setMultiplanarLayout(2);
        viewer.setMultiplanarPadPixels(8);

        // Slightly zoom the anatomical slices.
        viewer.setPan2Dxyzmm([0, 0, 0, 1.05]);
        // Improve 3D rendering size.
        viewer.setScale(1.1);
        // Display anatomical orientation markers.
        viewer.setIsOrientationTextVisible(true);
        viewer.setShowAllOrientationMarkers(true);

        // Configure synchronized crosshair appearance.
        viewer.setCrosshairColor([0.2, 0.85, 1, 0.9]);
        viewer.setCrosshairWidth(1);

        // Configure segmentation overlay.
        viewer.setOpacity(1, showMask ? opacity : 0);

        viewer.drawScene();

        setMessage(
          "MRI and segmentation loaded. Navigate the three views using the mouse and crosshair."
        );
      } catch (error) {
        console.error("Niivue loading error:", error);

        setMessage(
          error instanceof Error
            ? `Could not load NIfTI files: ${error.message}`
            : "Could not load the selected NIfTI files."
        );
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadViewer();

    return () => {
      cancelled = true;

      if (nv) {
        try {
          nv.cleanup();
        } catch {
          // Ignore cleanup errors.
        }
      }

      if (niivueRef.current === nv) {
        niivueRef.current = null;
      }

      if (imageUrl) URL.revokeObjectURL(imageUrl);
      if (maskUrl) URL.revokeObjectURL(maskUrl);
    };
  }, [image, mask]);

  // Update segmentation opacity without reloading the volumes.
  useEffect(() => {
    const nv = niivueRef.current;

    if (!nv?.volumes?.[1]) return;

    nv.setOpacity(1, showMask ? opacity : 0);
    nv.drawScene();
  }, [opacity, showMask]);
  
  // Update individual segmentation label visibility.
  useEffect(() => {
    const nv: any = niivueRef.current;

    if (!nv?.volumes?.[1]) return;

    const segmentation: any = nv.volumes[1];

    // Colors for known labels.
    const palette: Record<number, number[]> = {
      1: [255, 80, 40],   // Coral
      2: [255, 180, 0],  // Amber
      3: [220, 50, 220], // Magenta
    };

    // Generate colors dynamically for the labels present.
    const colors = labelStats.map(
      (item) => palette[item.label] ?? [80, 200, 255]
    );

    // Rebuild the discrete label color map.
    segmentation.setColormapLabel({
      R: [0, ...colors.map((c) => c[0])],
      G: [0, ...colors.map((c) => c[1])],
      B: [0, ...colors.map((c) => c[2])],

      // Background is transparent.
      // Hidden labels also become transparent.
      A: [
        0,
        ...labelStats.map((item) =>
          visibleLabels[item.label] === false ? 0 : 255
        ),
      ],

      I: [0, ...labelStats.map((item) => item.label)],

      labels: [
        "Background",
        ...labelStats.map((item) => `Label ${item.label}`),
      ],
    });

    nv.drawScene();
  }, [labelStats, visibleLabels]);

  // Validate selected NIfTI files.
  function validNifti(file: File) {
    return /\.nii(\.gz)?$/i.test(file.name);
  }

  function selectFile(file: File | undefined, kind: "image" | "mask") {
    if (!file) return;

    if (!validNifti(file)) {
      setMessage("Please select a .nii or .nii.gz file.");
      return;
    }

    if (kind === "image") {
      setImage(file);
    } else {
      setMask(file);
    }

    setMessage(
      "File selected. Make sure the MRI and segmentation belong to the same patient."
    );
  }

  return (
    <section className="viewerSection" id="mri-viewer">
      <div className="sectionHead">
        <div>
          <div className="eyebrow muted">WORKSPACE / 03</div>
          <h2>Interactive MRI viewer</h2>
        </div>

        <div className="formatPill">NIfTI · Local preview</div>
      </div>

      <p className="sectionIntro">
        Explore MRI volumes and their corresponding segmentation masks using
        synchronized axial, coronal and sagittal views. Files are opened
        locally in your browser and are not uploaded by this viewer.
      </p>

      {/* File selection */}
      <div className="viewerInputs">
        <label className="viewerFile">
          <span>01 · Original MRI</span>

          <input
            type="file"
            accept=".nii,.nii.gz"
            onChange={(e) =>
              selectFile(e.target.files?.[0], "image")
            }
          />

          <strong>
            {image?.name || "Choose MRI (.nii / .nii.gz)"}
          </strong>
        </label>

        <label className="viewerFile">
          <span>02 · Segmentation mask</span>

          <input
            type="file"
            accept=".nii,.nii.gz"
            onChange={(e) =>
              selectFile(e.target.files?.[0], "mask")
            }
          />

          <strong>
            {mask?.name || "Choose mask (.nii / .nii.gz)"}
          </strong>
        </label>
      </div>

      {/* Viewer controls */}
      <div className="viewerToolbar">
        <div className="planeButtons">
          <span className="viewerModeLabel">
            MULTIPLANAR VIEW
          </span>

          <span className="formatPill">
            Axial · Coronal · Sagittal
          </span>
        </div>

        <label className="opacityControl">
          <span>Mask opacity</span>

          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            disabled={!showMask}
            onChange={(e) =>
              setOpacity(Number(e.target.value))
            }
          />

          <b>{Math.round(opacity * 100)}%</b>
        </label>

        <button
          type="button"
          className={showMask ? "planeButton active" : "planeButton"}
          onClick={() => setShowMask((previous) => !previous)}
        >
          {showMask ? "Hide mask" : "Show mask"}
        </button>
      </div>

      {/* MRI canvas */}
      <div className="viewerCanvasWrap">
        <canvas
          ref={canvasRef}
          className="niivueCanvas"
          width={1400}
          height={1000}
          aria-label="Interactive multiplanar MRI and segmentation viewer"
        />

        {(!image || !mask) && (
          <div className="viewerEmpty">
            <div className="viewerGlyph">◉</div>

            <strong>Load two matching NIfTI volumes</strong>

            <span>Original MRI + segmentation mask</span>

            <span className="viewerEmptyHint">
              Three anatomical views will appear here.
            </span>
          </div>
        )}

        {loading && (
          <div className="viewerLoading">
            Loading MRI volumes...
          </div>
        )}
      </div>

      <p className="viewerMessage" role="status">
        {message}
      </p>
      {/* Segmentation analytics */}
      {labelStats.length > 0 && (
        <div className="viewerAnalytics">

          <h3>Segmentation analytics</h3>

          <div className="analyticsSummary">
            <div>
              <span>Total segmented volume</span>
              <strong>
                {totalVolumeCm3.toFixed(3)} cm³
              </strong>
            </div>

            <div>
              <span>Segmented voxels</span>
              <strong>
                {labelStats.reduce(
                  (sum, item) => sum + item.count,
                  0
                ).toLocaleString()}
              </strong>
            </div>
          </div>

          <h4>Label-wise statistics and visibility</h4>

          {labelStats.map((item) => (
            <div className="analyticsRow" key={item.label}>

              <label>
                <input
                  type="checkbox"
                  checked={visibleLabels[item.label] ?? true}
                  onChange={(e) => {
                    setVisibleLabels((previous) => ({
                      ...previous,
                      [item.label]: e.target.checked,
                    }));
                  }}
                />

                Label {item.label}
              </label>

              <span>
                {item.count.toLocaleString()} voxels
              </span>

              <span>
                {item.volumeCm3.toFixed(3)} cm³
              </span>

              <span>
                {item.percentage.toFixed(2)}%
              </span>

            </div>
          ))}

        </div>
      )}

      {/* Legend */}
      <div className="viewerLegend">
        <span>
          <i className="legendMri" />
          MRI intensity
        </span>

        <span>
          <i className="legendMask" />
          Segmentation overlay
        </span>

        <span>
          <i
            style={{
              display: "inline-block",
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: "#33d9ff",
              marginRight: 7,
            }}
          />
          Crosshair
        </span>
      </div>

      <div className="viewerWarning">
        <strong>Alignment requirement:</strong> The segmentation must
        belong to the same patient and spatial reference as the displayed
        MRI. Confirm dimensions, affine, orientation and voxel spacing
        before interpreting an overlay.

        This viewer is intended for research demonstration, not clinical
        diagnosis.
      </div>
    </section>
  );
}