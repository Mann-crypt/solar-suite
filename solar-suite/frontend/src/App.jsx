import React, {
  useState
} from "react";

import {
  CloudSun,
  Upload,
  BarChart3,
  Settings,
  Activity,
  FileSpreadsheet,
  RefreshCw,
  Download,
  CheckCircle2,
  AlertCircle,
  Loader2
} from "lucide-react";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from "recharts";

import {
  runFixed,
  runTracking,
  recalcTracking,
  rtRecalculate,
  uploadFile,
  pollJob
} from "./api";


function App() {

  const [page, setPage] =
    useState("loss");

  return (
    <div className="app">

      <Sidebar
        page={page}
        setPage={setPage}
      />

      <main className="main">

        <TopBar />

        {page === "loss" && (
          <LossCorrection />
        )}

        {page === "rt" && (
          <RTCorrection />
        )}

        {page === "aeromal" && (
          <div className="placeholder">
            AeroMal module coming soon.
          </div>
        )}

      </main>

    </div>
  );
}


// ==================================================
// SIDEBAR
// ==================================================

function Sidebar({
  page,
  setPage
}) {

  return (
    <aside className="sidebar">

      <div className="brand">

        <div className="brand-icon">
          <CloudSun size={24} />
        </div>

        <div>
          <div className="brand-title">
            Solar Suite
          </div>

          <div className="brand-subtitle">
            Forecasting Platform
          </div>
        </div>

      </div>


      <div className="nav-label">
        MODULES
      </div>


      <button
        className={
          page === "loss"
            ? "nav-item active"
            : "nav-item"
        }
        onClick={() => setPage("loss")}
      >
        <Activity size={18} />
        Loss Correction
      </button>


      <button
        className={
          page === "rt"
            ? "nav-item active"
            : "nav-item"
        }
        onClick={() => setPage("rt")}
      >
        <BarChart3 size={18} />
        RT Correction
      </button>


      <button
        className={
          page === "aeromal"
            ? "nav-item active"
            : "nav-item"
        }
        onClick={() => setPage("aeromal")}
      >
        <Settings size={18} />
        AeroMal
      </button>


      <div className="sidebar-bottom">

        <div className="system-status">
          <span className="status-dot"></span>

          Backend connected
        </div>

      </div>

    </aside>
  );
}


// ==================================================
// TOP BAR
// ==================================================

function TopBar() {

  return (
    <header className="topbar">

      <div>
        <div className="topbar-title">
          Solar Analytics
        </div>

        <div className="topbar-subtitle">
          Solar forecasting and performance correction
        </div>
      </div>


      <div className="topbar-right">

        <div className="online">
          <span></span>
          System ready
        </div>

      </div>

    </header>
  );
}


// ==================================================
// LOSS CORRECTION
// ==================================================

function LossCorrection() {

  const [file, setFile] =
    useState(null);

  const [fileId, setFileId] =
    useState(null);

  const [plantType, setPlantType] =
    useState("Fixed");

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [uploading, setUploading] =
    useState(false);

  const [error, setError] =
    useState("");

  const [progress, setProgress] =
    useState(0);


  // -----------------------------------------------
  // FILE UPLOAD
  // -----------------------------------------------

  async function handleFileUpload(
    selectedFile
  ) {

    if (!selectedFile) return;

    setFile(selectedFile);
    setError("");
    setResult(null);
    setUploading(true);

    try {

      const response =
        await uploadFile(selectedFile);

      setFileId(response.file_id);

    } catch (err) {

      setError(
        err.message
      );

    } finally {

      setUploading(false);

    }
  }


  // -----------------------------------------------
  // RUN
  // -----------------------------------------------

  async function handleRun() {

    if (!fileId) {

      setError(
        "Please upload an Excel file first."
      );

      return;
    }

    setLoading(true);
    setError("");
    setProgress(0);

    try {

      let response;

      if (plantType === "Fixed") {

        response =
          await runFixed(fileId);

      } else {

        response =
          await runTracking(fileId);

        if (response.job_id) {

          response =
            await pollJob(
              response.job_id,
              setProgress
            );

        }

      }

      setResult(response);

    } catch (err) {

      setError(
        err.message
      );

    } finally {

      setLoading(false);

    }
  }


  return (
    <div className="page">

      <div className="page-heading">

        <div>

          <h1>
            Loss Correction
          </h1>

          <p>
            Correct forecast losses using plant-specific
            performance parameters.
          </p>

        </div>

      </div>


      {/* ------------------------------------------ */}
      {/* INPUT CARD */}
      {/* ------------------------------------------ */}

      <section className="card">

        <div className="card-header">

          <div>

            <h2>
              Input Data
            </h2>

            <p>
              Upload your Excel file containing
              GHI Forecast and Actual data.
            </p>

          </div>

        </div>


        <div className="input-grid">

          <div>

            <label>
              Excel File
            </label>

            <label className="upload-box">

              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={e =>
                  handleFileUpload(
                    e.target.files?.[0]
                  )
                }
              />

              <Upload size={28} />

              {uploading ? (
                <span>
                  Uploading...
                </span>
              ) : file ? (
                <span>
                  {file.name}
                </span>
              ) : (
                <>
                  <strong>
                    Drop your file here
                  </strong>

                  <small>
                    XLSX, XLS or CSV
                  </small>
                </>
              )}

            </label>

          </div>


          <div>

            <label>
              Plant Type
            </label>

            <div className="segmented">

              <button
                className={
                  plantType === "Fixed"
                    ? "selected"
                    : ""
                }
                onClick={() =>
                  setPlantType("Fixed")
                }
              >
                Fixed
              </button>

              <button
                className={
                  plantType === "Tracking"
                    ? "selected"
                    : ""
                }
                onClick={() =>
                  setPlantType("Tracking")
                }
              >
                Tracking
              </button>

            </div>

          </div>

        </div>


        {fileId && (

          <div className="file-success">

            <CheckCircle2 size={18} />

            File uploaded successfully

          </div>

        )}


        {error && (

          <div className="error-box">

            <AlertCircle size={18} />

            {error}

          </div>

        )}


        <div className="run-row">

          <button
            className="primary-button"
            onClick={handleRun}
            disabled={
              !fileId ||
              loading ||
              uploading
            }
          >

            {loading ? (

              <>
                <Loader2
                  size={18}
                  className="spin"
                />

                Running...
              </>

            ) : (

              <>
                <Activity size={18} />

                Run Correction
              </>

            )}

          </button>

        </div>


        {loading &&
          plantType === "Tracking" && (

          <div className="progress-wrapper">

            <div className="progress-header">

              <span>
                Optimizing parameters
              </span>

              <span>
                {Math.round(progress)}%
              </span>

            </div>

            <div className="progress-bar">

              <div
                style={{
                  width: `${progress}%`
                }}
              />

            </div>

          </div>

        )}

      </section>


      {/* ------------------------------------------ */}
      {/* RESULT */}
      {/* ------------------------------------------ */}

      {result && (

        <LossCorrectionResult
          result={result}
          plantType={plantType}
          fileId={fileId}
        />

      )}

    </div>
  );
}


// ==================================================
// RESULT
// ==================================================

function LossCorrectionResult({
  result,
  plantType,
  fileId
}) {

  const chartData =
    buildChartData(result.chart);


  return (
    <>

      {/* METRICS */}

      <section className="metrics">

        <MetricCard
          title="Best Loss"
          value={
            result.best_loss != null
              ? `${Number(
                  result.best_loss
                ).toFixed(2)}%`
              : "--"
          }
        />

        <MetricCard
          title="MAE"
          value={
            result.metrics?.MAE != null
              ? Number(
                  result.metrics.MAE
                ).toFixed(3)
              : "--"
          }
        />

        <MetricCard
          title="RMSE"
          value={
            result.metrics?.RMSE != null
              ? Number(
                  result.metrics.RMSE
                ).toFixed(3)
              : "--"
          }
        />

        <MetricCard
          title="R²"
          value={
            result.metrics?.R2 != null
              ? Number(
                  result.metrics.R2
                ).toFixed(4)
              : "--"
          }
        />

      </section>


      {/* CHART */}

      {chartData.length > 0 && (

        <section className="card chart-card">

          <div className="card-header">

            <div>

              <h2>
                Forecast Performance
              </h2>

              <p>
                Actual vs forecast after loss correction
              </p>

            </div>

          </div>


          <div className="chart-container">

            <ResponsiveContainer
              width="100%"
              height={430}
            >

              <LineChart
                data={chartData}
              >

                <CartesianGrid
                  strokeDasharray="3 3"
                />

                <XAxis
                  dataKey="block"
                />

                <YAxis />

                <Tooltip />

                <Legend />

                <Line
                  type="monotone"
                  dataKey="forecast"
                  name="Forecast"
                  strokeWidth={2}
                  dot={false}
                />

                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Actual"
                  strokeWidth={2}
                  dot={false}
                />

                {result.chart.corrected && (

                  <Line
                    type="monotone"
                    dataKey="corrected"
                    name="Corrected"
                    strokeWidth={3}
                    dot={false}
                  />

                )}

              </LineChart>

            </ResponsiveContainer>

          </div>

        </section>

      )}


      {/* PARAMETERS */}

      {result.parameters && (

        <section className="card">

          <div className="card-header">

            <div>

              <h2>
                Optimized Parameters
              </h2>

              <p>
                Parameters selected by the optimizer.
              </p>

            </div>

          </div>


          <div className="parameter-grid">

            {Object.entries(
              result.parameters
            ).map(
              ([key, value]) => (

                <div
                  className="parameter"
                  key={key}
                >

                  <span>
                    {formatParameterName(key)}
                  </span>

                  <strong>
                    {typeof value === "number"
                      ? value.toFixed(3)
                      : value}
                  </strong>

                </div>

              )
            )}

          </div>

        </section>

      )}


      {/* EFFICIENCY */}

      {result.efficiency_table && (

        <section className="card">

          <div className="card-header">

            <div>

              <h2>
                Efficiency Summary
              </h2>

              <p>
                Calculated efficiency and loss values.
              </p>

            </div>

            <button className="secondary-button">

              <Download size={16} />

              Download

            </button>

          </div>


          <DataTable
            data={
              result.efficiency_table
            }
          />

        </section>

      )}

    </>
  );
}


// ==================================================
// METRIC CARD
// ==================================================

function MetricCard({
  title,
  value
}) {

  return (

    <div className="metric-card">

      <span>
        {title}
      </span>

      <strong>
        {value}
      </strong>

    </div>

  );
}


// ==================================================
// DATA TABLE
// ==================================================

function DataTable({
  data
}) {

  if (!data) return null;

  const rows =
    Array.isArray(data)
      ? data
      : data.rows || [];

  if (!rows.length) {

    return (
      <div className="empty-table">
        No table data available.
      </div>
    );

  }

  const columns =
    Object.keys(rows[0]);

  return (

    <div className="table-wrapper">

      <table>

        <thead>

          <tr>

            {columns.map(
              column => (
                <th key={column}>
                  {formatParameterName(
                    column
                  )}
                </th>
              )
            )}

          </tr>

        </thead>

        <tbody>

          {rows.map(
            (row, index) => (

              <tr key={index}>

                {columns.map(
                  column => (

                    <td key={column}>

                      {formatTableValue(
                        row[column]
                      )}

                    </td>

                  )
                )}

              </tr>

            )
          )}

        </tbody>

      </table>

    </div>

  );
}


// ==================================================
// RT PAGE
// ==================================================

function RTCorrection() {

  return (

    <div className="page">

      <div className="page-heading">

        <div>

          <h1>
            RT Correction
          </h1>

          <p>
            Real-time forecast curve correction.
          </p>

        </div>

      </div>


      <section className="card">

        <div className="placeholder-content">

          <BarChart3 size={40} />

          <h2>
            RT Correction
          </h2>

          <p>
            Upload your data to start RT correction.
          </p>

        </div>

      </section>

    </div>

  );
}


// ==================================================
// HELPERS
// ==================================================

function buildChartData(chart) {

  if (!chart) return [];

  const x =
    chart.x || [];

  const forecast =
    chart.forecast || [];

  const actual =
    chart.actual || [];

  const corrected =
    chart.corrected || [];

  return x.map(
    (block, index) => ({

      block,

      forecast:
        forecast[index],

      actual:
        actual[index],

      ...(corrected.length
        ? {
            corrected:
              corrected[index]
          }
        : {})

    })
  );
}


function formatParameterName(
  value
) {

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, char =>
      char.toUpperCase()
    );
}


function formatTableValue(
  value
) {

  if (
    typeof value === "number"
  ) {

    return Number(value)
      .toFixed(3);

  }

  return value ?? "--";
}


export default App;
