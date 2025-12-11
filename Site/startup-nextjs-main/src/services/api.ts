// Dynamically construct API URL using current hostname but backend port 8000
const getApiBaseUrl = (): string => {
    if (typeof window !== "undefined") {
        const hostname = window.location.hostname;
        return `http://${hostname}:8000/api`;
    }
    return "http://localhost:8000/api";
};

export async function analyzeDocument(file: File, metadata?: Record<string, any>) {
    const API_URL = getApiBaseUrl();
    const formData = new FormData();
    formData.append("file", file);

    // Append metadata as JSON string if provided
    if (metadata) {
        formData.append("metadata", JSON.stringify(metadata));
    }

    console.log(`Calling API at: ${API_URL}/analyze`);
    console.log("With metadata:", metadata);

    const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error("API Error:", errorText);
        throw new Error(`Analysis failed: ${response.statusText}`);
    }

    return response.json();
}
