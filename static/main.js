// Update Settings with Data and DB Directory
function updateSettings() {
  const chunkSize = document.getElementById("chunkSize").value;
  const embeddingModel = document.getElementById("embeddingModel").value;
  const dataDir = document.getElementById("dataDir").value;
  const dbDir = document.getElementById("dbDir").value;

  const newSettings = {
    chunk_size: parseInt(chunkSize),
    embedding_model: embeddingModel,
    data_dir: dataDir,
    db_dir: dbDir,
  };

  fetch("/update-settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(newSettings),
  })
    .then((response) => response.json())
    .then((data) => alert(data.message))
    .catch((err) => console.error("Error updating settings:", err));
}

// File Upload
function uploadFile() {
  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];
  if (!file) return alert("Please select a file.");

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload-file", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("uploadStatus").innerText =
        data.message || data.error;
      if (data.message) window.location.reload();
    })
    .catch((err) => console.error("File upload error:", err));
}

// File Delete
function deleteFile(filename) {
  fetch("/delete-file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  })
    .then((response) => response.json())
    .then((data) => {
      alert(data.message || data.error);
      if (data.message) window.location.reload();
    })
    .catch((err) => console.error("File delete error:", err));
}
