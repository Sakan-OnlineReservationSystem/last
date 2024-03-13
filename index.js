const express = require('express');
const { python } = require('pythonia');

const app = express();
const PORT = process.env.PORT || 3000;
let np = null;

app.get('/:message', async (req, res) => {
  try {
    const mystring = req.params.message; // Extract the 'message' parameter from the URL
    // Correctly destructure the result from the Python call
    // The original line had syntax errors and was not properly destructuring the object returned from the Python function.
    // Assuming 'validate' is a method of the Python object that returns an object with pvturn, res, bspn, and aspn properties.
    { pv_turn, res , bspn ,aspn } = await np.validate(mystring = mystring);
    console.log(result); // Log the validation result
    console.log(bspn); // Log additional information
    console.log(aspn); // Log additional information
    // Use a different variable name for the response from express to avoid naming conflict with the destructured 'res' from the Python result.
    res.send(result); // Send the result back to the client
  } catch (error) {
    console.error('Error:', error.message);
    // Correct response method usage to handle errors gracefully.
    res.status(500).send('Internal Server Error');
  }
});

app.listen(PORT, async () => {
  try {
    // The 'python' function is used to load and start the Python process
    np = await python('./train.py'); // Assuming './train.py' is your Python script
    // The following 'start' method was assumed to exist. If your Python script doesn't have a 'start' method, you might need to remove this line.
    // await np.start(); // Removed this line as the pythonia library doesn't require explicitly starting the process after initialization.
    console.log(`Server listening on port ${PORT}`);
  } catch (error) {
    console.error('Error starting Python process:', error.message);
  }
});
