const express = require('express');
const { python } = require('pythonia');

const app = express();
const PORT = process.env.PORT || 3000;
let np = null;

app.get('/:message', async (req, res) => {
  try {
    const myString = req.params.message; // Good practice to use camelCase in JavaScript
    // Assuming 'validate' is a method of the Python object that directly accepts the message argument
    let result = await np.validate(myString);
    console.log(result.res); // If result is an object and res a property, no need for await
    res.send(result); // Send the result back to the client
  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).send('Internal Server Error');
  }
});

app.listen(PORT, async () => {
  try {
    np = await python('./train.py'); // Assuming './train.py' is your Python script and it exports the necessary functionality
    console.log(`Server listening on port ${PORT}`);
  } catch (error) {
    console.error('Error starting Python process:', error.message);
  }
});
