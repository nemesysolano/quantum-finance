from .scaling import scale_features, scale_targets
from .trainers import meta_trainer, base_trainer
from .models import base
base_model_factories = {
    'prob': base.probdiff,
    'pricevol': base.pricevoldiff,
    'priceangle': base.priceangle,
    'gauge': base.gauge
}