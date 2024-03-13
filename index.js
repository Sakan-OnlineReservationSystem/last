const express = require('express');
const { py, python } = require('pythonia');

const app = express();
const PORT = process.env.PORT || 3000;
let np = null;

app.get('/:message', async (req, res) => {
  try {
    const mystring = req.params.message; // Extract the 'message' parameter from the URL
    const pvturn, res ,bspn , aspn = await np.validate(mystring); // Assuming 'validate' is a method in your Python class
    console.log(res); // Log the validation result
    res.send(result); // Send the result back to the client
  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).send('Internal Server Error'); // Handle errors gracefully
  }
});

app.listen(PORT, async () => {
  try {
    np = await python('./train.py'); // Initialize your Python process
    await np.start(); // Start the Python process
    console.log(`Server listening on port ${PORT}`);
  } catch (error) {
    console.error('Error starting Python process:', error.message);
  }
});
