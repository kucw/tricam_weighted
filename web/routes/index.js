var exec = require('child_process').exec;
var fs = require('fs');

module.exports = function(app){	
	app.get('/', function(req ,res){
		res.render('index', {	
			mode:"mode",
		});
	});
	app.get('/fps', function(req ,res){
		res.render('fps', {	
			mode:"mode",
		});
	});


    app.post('/', function(req, res) {
		var pyexec = require('python-shell');
		var options = {
			mode: "text",
		};
		pyexec.run('./python/monitor.py', options, function (err, result) {
			//console.log(result);
			//console.log(JSON.parse(result));
			var gpu_u = JSON.parse(result).gpu_utilization;
			var gpu_mem_u = JSON.parse(result).gpu_memory_utilization;
			var cpu_u = JSON.parse(result).cpu_utilization;
			var cpu_mem_u = JSON.parse(result).cpu_memory_utilization;
			res.send({
				gpu_util: gpu_u,
				gpu_mem_util: gpu_mem_u,
				cpu_util: cpu_u,
				cpu_mem_util: cpu_mem_u,
			});

		});
	});
	app.post('/fps', function(req, res) {
		exec('ps x -o pid,command | grep darknet | grep yolo | grep -v grep', function(err, stdout, stderr) {
		var pids = []
			stdout.split("\n").forEach(function(v) {
				var arr = v.trim().split(" ");
					if (arr.length == 1) {
						return;
					}
				pids.push({
					pid: arr[0],
					cmd: arr.slice(1).join(" "),
				});
			});
		if (pids.length == 0) {
	        //console.log("cannot find yolo");
			res.send({
				fps: 0,
			});
	    }
		else {
			console.log(pids[0].pid);
			process.kill(pids[0].pid, "SIGUSR2");
			var path = "../fps.txt";
			fs.open(path, "r", function(err, fd) {
				var buf = new Buffer(4);
				fs.readSync(fd, buf, 0, 4);
				fs.closeSync(fd);
				res.send({
					fps: buf.readInt32BE(0),	
				});
			});
		}
    });	  
			
});
	/*
	app.get('/', function(req ,res){
		res.render('index');
		var exec = require('python-shell');
		var options = {
		};
		exec.run('./python/transaction.py', options, function(err, result){
			if(err) throw err;
			var tx_hash = JSON.parse(result).tx_hash;
			data1.to.push({address: to_address, hash: tx_hash});
			data1.save(function(err){
				if(err) console.error(err);
			});
			data2.from.push({address: from_address, hash: tx_hash});
			data2.save(function(err){
				if(err) console.error(err);
			});
		});

	}
	*/
};
