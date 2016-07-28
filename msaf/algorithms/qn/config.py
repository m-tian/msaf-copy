"""Config for the Quadratic novelty algorithm."""

# Params
config = {
	"alpha": 		9.0, # Alpha norm
	"delta": 		0.0, # Adaptive thresholding delta
	"rawSensitivity": 80,
	"preWin": 		10,
	"postWin": 		10,
	"LP_on": 		True,
	"Medfilt_on": 	True,
	"Polyfit_on": 	True,
	"isMedianPositive": False,
	"kernel_size":	100
}

algo_id = "qn" 
is_boundary_type = True
is_label_type = False
