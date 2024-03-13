const express = require('express');
const { py, python } = require('pythonia')

const app = express();
const PORT = process.env.PORT || 3000;
let np = null ;

app.get('/', async (req, res) => {
  console.log(await np.plus())
});

app.listen(PORT, async () => {
  np = await python("./test.py")
  console.log(`Server listening on port ${PORT}`);
});
