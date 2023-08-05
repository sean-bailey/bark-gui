import os

def initenv(args):
    os.environ['SUNO_USE_SMALL_MODELS'] = "True"#str("-smallmodels" in args)
    os.environ['BARK_FORCE_CPU'] = "True"#str("-forcecpu" in args)
    os.environ['SUNO_ENABLE_MPS'] = "False"#str("-enablemps" in args)
    os.environ['SUNO_OFFLOAD_CPU'] = "True"#str("-offloadcpu" in args)
