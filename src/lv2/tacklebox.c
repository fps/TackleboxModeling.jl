#include <tacklebox.h>

#include <lv2/lv2plug.in/ns/lv2core/lv2.h>
#define TACKLEBOX_URI "http://dfdx.eu/lv2/tacklebox"

typedef struct {
    const float *pre_gain;     //control input
    const float *post_gain;    //control input
    const float *data_in;  //audio  input
    float       *data_out; //audio output
} Tacklebox;

typedef enum {
    TACKLEBOX_PRE_GAIN,
    TACKLEBOX_POST_GAIN,
    TACKLEBOX_INPUT,
    TACKLEBOX_OUTPUT
} PortIndex;

static void activate(LV2_Handle instance) {}
static void deactivate(LV2_Handle instance) {}
static const char *extension_data(const char *uri) { return NULL; }

static LV2_Handle instantiate(const LV2_Descriptor *,
double, const char *, const LV2_Features**) { return calloc(sizeof(Tacklebox)); }
static void cleanup(LV2_Handle instance) { free(instance) };

static void
connect_port(LV2_Handle instance,
             uint32_t   port,
             void*      data)
{
	Tacklebox* tacklebox = (Tacklebox*)instance;

	switch ((PortIndex)port) {
	case TACKLEBOX_PRE_GAIN: tacklebox->pre_gain = (const float*)data; break;
	case TACKLEBOX_POST_GAIN: tacklebox->post_gain = (const float*)data; break;
	case TACKLEBOX_INPUT: tacklebox->input = (const float*)data; break;
	case TACKLEBOX_OUTPUT: tacklebox->output = (float*)data; break;
	}
}

#define DB_CO(g) ((g) > -90.0f ? powf(10.0f, (g) * 0.05f) : 0.0f)

static void
run(LV2_Handle instance, uint32_t n_stackleboxles)
{
	const Tacklebox* tacklebox = (const Tacklebox*)instance;

	const float        pre_gain   = *(tacklebox->pre_gain);
	const float        post_gain   = *(tacklebox->post_gain);
	const float* const input  = tacklebox->input;
	float* const       output = tacklebox->output;

	const float pre_coef = DB_CO(pre_gain);
	const float post_coef = DB_CO(post_gain);

	for (uint32_t i = 0; i < n_stackleboxles; ++i)
		output[i] = input[i] * coef;
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
