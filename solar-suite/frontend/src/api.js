const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "";

async function handleResponse(response) {
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data.detail ||
      data.error ||
      "Something went wrong."
    );
  }

  return data;
}


// --------------------------------------------------
// FIXED
// --------------------------------------------------

export async function runFixed(
  fileId,
  editedRows = null
) {
  const form = new FormData();

  form.append("file_id", fileId);

  if (editedRows) {
    form.append(
      "edited_rows",
      JSON.stringify(editedRows)
    );
  }

  const response = await fetch(
    `${API_BASE}/api/loss-correction/fixed`,
    {
      method: "POST",
      body: form
    }
  );

  return handleResponse(response);
}


// --------------------------------------------------
// TRACKING OPTIMIZATION
// --------------------------------------------------

export async function runTracking(
  fileId,
  editedRows = null
) {
  const form = new FormData();

  form.append("file_id", fileId);

  if (editedRows) {
    form.append(
      "edited_rows",
      JSON.stringify(editedRows)
    );
  }

  const response = await fetch(
    `${API_BASE}/api/loss-correction/tracking`,
    {
      method: "POST",
      body: form
    }
  );

  return handleResponse(response);
}


// --------------------------------------------------
// TRACKING RECALCULATE
// --------------------------------------------------

export async function recalcTracking(
  fileId,
  params,
  editedRows = null
) {
  const form = new FormData();

  form.append("file_id", fileId);

  form.append(
    "params",
    JSON.stringify(params)
  );

  if (editedRows) {
    form.append(
      "edited_rows",
      JSON.stringify(editedRows)
    );
  }

  const response = await fetch(
    `${API_BASE}/api/loss-correction/tracking/recalculate`,
    {
      method: "POST",
      body: form
    }
  );

  return handleResponse(response);
}


// --------------------------------------------------
// RT RECALCULATE
// --------------------------------------------------

export async function rtRecalculate(
  actual,
  trend,
  params
) {
  const form = new FormData();

  form.append(
    "actual",
    JSON.stringify(actual)
  );

  form.append(
    "trend",
    JSON.stringify(trend)
  );

  form.append(
    "params",
    JSON.stringify(params)
  );

  const response = await fetch(
    `${API_BASE}/api/rt-correction/recalculate`,
    {
      method: "POST",
      body: form
    }
  );

  return handleResponse(response);
}


// --------------------------------------------------
// JOB STATUS
// --------------------------------------------------

export async function getJobStatus(jobId) {

  const response = await fetch(
    `${API_BASE}/api/status/${jobId}`
  );

  return handleResponse(response);
}


export async function pollJob(
  jobId,
  onProgress
) {

  while (true) {

    const job =
      await getJobStatus(jobId);

    if (onProgress && job.progress != null) {
      onProgress(job.progress);
    }

    if (job.status === "done") {
      return job.result;
    }

    if (job.status === "error") {
      throw new Error(
        job.error || "Optimization failed."
      );
    }

    await new Promise(
      resolve => setTimeout(resolve, 800)
    );
  }
}


// --------------------------------------------------
// FILE UPLOAD
// --------------------------------------------------

export async function uploadFile(file) {

  const form = new FormData();

  form.append("file", file);

  const response = await fetch(
    `${API_BASE}/api/upload`,
    {
      method: "POST",
      body: form
    }
  );

  return handleResponse(response);
}
