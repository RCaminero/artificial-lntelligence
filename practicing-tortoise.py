# Basic operations with single model
from tortoise import Tortoise, fields, run_async
from tortoise.models import Model
import asyncio
import nest_asyncio
nest_asyncio.apply()


class Event(Model):
    id = fields.IntField(pk=True)
    name = fields.TextField()
    capital = fields.TextField(null=True)

    class Meta:
        table = "countries"

    def __str__(self):
        return self.name


async def run():
    await Tortoise.init(db_url="sqlite://:memory:", modules={"models": ["__main__"]})
    await Tortoise.generate_schemas()

    event = await Event.create(name="countryTest", capital="capitalTest")

    # Save data
    await Event(name="Dominican Republic", capital="Santo Domingo").save()
    await Event(name="Austria", capital="Vienna").save()
    await Event(name="Haiti", capital="Pourt Prince").save()

    # Update Date
    await Event.filter(id=3).update(name="Did it")

    # 

    print(await Event.all().values_list("name", flat=True))
    # print(await Event.filter(id=2).first())
    # # >>> [1, 2]
    # print(await Event.all().values("id", "name"))
    # # >>> [{'id': 1, 'name': 'Updated name'}, {'id': 2, 'name': 'Test 2'}]


if __name__ == "__main__":
    run_async(run())