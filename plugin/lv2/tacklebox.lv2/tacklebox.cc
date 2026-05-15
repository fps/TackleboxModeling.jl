#include <tacklebox.h>
#include <lv2/core/lv2.h>
#include <iostream>

#define TACKLEBOX_URI "http://dfdx.eu/lv2/tacklebox"

std::vector<tacklebox::model> models =
{
  #include "../../../data/BrianMay/model.cc"
};

struct Tacklebox{
    tacklebox::processor *t;
    const float *pre_gain;     //control input
    const float *post_gain;    //control input
    const float *data_in;  //audio  input
    float       *data_out; //audio output

    Tacklebox() :
      t(new tacklebox::processor(models[0], 64))
    {

    }

    ~Tacklebox()
    {
      delete t;
    }
};

typedef enum {
    TACKLEBOX_PRE_GAIN,
    TACKLEBOX_POST_GAIN,
    TACKLEBOX_INPUT,
    TACKLEBOX_OUTPUT
} PortIndex;

static void activate(LV2_Handle instance) {}
static void deactivate(LV2_Handle instance) {}
static const void *extension_data(const char *uri) { return NULL; }

static LV2_Handle instantiate(const LV2_Descriptor * d, double x, const char *c, const LV2_Feature *const *f) 
{ 
    std::cout << "instantiate...\n";
  return (LV2_Handle) new Tacklebox; }
static void cleanup(LV2_Handle instance) { delete (Tacklebox*)instance; }

static void
connect_port(LV2_Handle instance,
             uint32_t   port,
             void*      data)
{
	Tacklebox* tacklebox = (Tacklebox*)instance;

	switch ((PortIndex)port) {
	case TACKLEBOX_PRE_GAIN: tacklebox->pre_gain = (const float*)data; break;
	case TACKLEBOX_POST_GAIN: tacklebox->post_gain = (const float*)data; break;
	case TACKLEBOX_INPUT: tacklebox->data_in = (const float*)data; break;
	case TACKLEBOX_OUTPUT: tacklebox->data_out = (float*)data; break;
	}
}

#define DB_CO(g) ((g) > -90.0f ? powf(10.0f, (g) * 0.05f) : 0.0f)

static void
run(LV2_Handle instance, uint32_t n_samples)
{
	const Tacklebox* tacklebox = (const Tacklebox*)instance;

	const float        pre_gain   = *(tacklebox->pre_gain);
	const float        post_gain   = *(tacklebox->post_gain);
	const float* const input  = tacklebox->data_in;
	float* const       output = tacklebox->data_out;

	const float pre_coef = DB_CO(pre_gain);
	const float post_coef = DB_CO(post_gain);

  int n_samples_left = n_samples;
  while (n_samples_left > 0)
  {
    if (n_samples_left >= 64)
    {
      tacklebox->t->process(input, output, pre_coef, post_coef, 64);
      n_samples_left -= 64;
    }
    else
    {
      tacklebox->t->process(input, output, pre_coef, post_coef, n_samples_left);
      n_samples_left -= n_samples_left;
    }
  }
}

static const LV2_Descriptor descriptor = {
	TACKLEBOX_URI,
	instantiate,
	connect_port,
	activate,
	run,
	deactivate,
	cleanup,
	extension_data
};

LV2_SYMBOL_EXPORT
const LV2_Descriptor*
lv2_descriptor(uint32_t index)
{
	switch (index) {
	case 0:  return &descriptor;
	default: return NULL;
	}
}

