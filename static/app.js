function displayResult(presence, category) {
    const resultElement = document.getElementById('result');
    resultElement.innerHTML = `
        <h3>Presence: ${presence}</h3>
        <h3>Category: ${category}</h3>
    `;
}

document.getElementById('submit').addEventListener('click', () => {
    const url = document.getElementById('url').value;

    if (!url) {
        alert('Please enter a URL');
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: url })
    })
    .then(response => response.json())
    .then(data => {
        displayResult(data.presence, data.category);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
