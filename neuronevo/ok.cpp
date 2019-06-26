#include "stdafx.h"
#include "ok.h"


ok::ok()
{
}

ok::ok(ok & O)
{
	i = O.i;
}

ok &ok::operator*(ok & r)
{
	ok result(*this);
	result.i = result.i*r.i;
	return result;
}


ok::~ok()
{
}

ok operator*(ok & r, ok & t)
{
	ok result(r);
	result.i = result.i*t.i;
	return result;

}
