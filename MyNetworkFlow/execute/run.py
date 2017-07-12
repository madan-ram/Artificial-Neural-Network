def run(opts, feed_dict={}):
	for k, v in feed_dict.items():
		k.value = v

	for o in opts:
		o.forward()
	return [o.value for o in opts]
