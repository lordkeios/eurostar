#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <cinttypes>
#include <cassert>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <xstring>
#include <set>
#include <map>
#include <deque>
#include <queue>
#include <vector>
#include <list>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <Windows.h>
#include <WinInet.h>
#include <tchar.h>
#include <time.h>
#include <atlcoll.h>
#include <atlconv.h>
#include <sql.h>
#include <sqlext.h>
#include <MMSystem.h>

#include <boost/any.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/function_output_iterator.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/pool/pool.hpp>
#include <boost/pool/object_pool.hpp>
#include <boost/pool/singleton_pool.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/function.hpp>
#include <boost/locale.hpp>
#include <boost/locale/conversion.hpp>
#include <boost/iterator/function_output_iterator.hpp>
#include <boost/format.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/insert_linebreaks.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/archive/iterators/ostream_iterator.hpp>
#include <boost/chrono.hpp>
#include <boost/program_options.hpp>

typedef boost::archive::iterators::insert_linebreaks<boost::archive::iterators::base64_from_binary<boost::archive::iterators::transform_width<const char *, 6, 8>>, 72> base64_text_linebreaks;
typedef boost::archive::iterators::base64_from_binary<boost::archive::iterators::transform_width<const char *, 6, 8>> base64_text;

FILE _iob[] = { *stdin, *stdout, *stderr };
extern "C" FILE * __cdecl __iob_func(void) { return _iob; }

#pragma comment(lib, "winmm")

