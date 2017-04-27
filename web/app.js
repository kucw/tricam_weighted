const express = require('express');
const path = require('path');
const app = express();
var routes = require('./routes/index');

// views middleware
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');
app.use(express.static(path.join(__dirname, 'public')));

routes(app);

app.listen('3000', '0.0.0.0', function(){
	console.log('Web server listening on port 3000')
});

