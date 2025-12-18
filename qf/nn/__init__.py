from .scaling import scale_features, scale_targets
from .models import base
base_model_factories = {
    'prob': base.probdiff,
    'pricevol': base.pricevoldiff,
    'wavelets': base.wavelets,
    'gauge': base.gauge
}