from app import create_app, consolidator, explorator
from app.models import *
from app.utils import cli

init_data = False
app = create_app()
cli.register(app)


@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'User': User,
        'UserContext': UserContext,
        'Context': Context,
        'Value': Value,
        'Motivation': Motivation,
        'Keyword': Keyword,
        'AnnotationAction': AnnotationAction,
        'ConsolidationAction': ConsolidationAction,
        'ConsolidationValue': ConsolidationValue,
        'ConsolidationKeyword': ConsolidationKeyword,
        'explorator': explorator,
        'consolidator': consolidator
    }


def init_ukraine_context(motivations_filename):
    from app.models import Context, Choice

    # write context to db
    context = Context(context_name_en='Ukraine Messages', context_name_nl='Ukraine Messages')
    db.session.add(context)
    db.session.commit()

    # Quick fix: arbitrary choices copied from covid context so that choice table is not empty,
    # not needed for ukraine context
    choices = [
        Choice(choice_order=1, choice_name_en='The municipality takes the lead and unburdens you',
               choice_name_nl='De gemeente neemt de leiding en ontzorgt', context_id=context.id),
        Choice(choice_order=2, choice_name_en='Inhabitants do it themselves',
               choice_name_nl='Inwoners doen het zelf', context_id=context.id),
        Choice(choice_order=3, choice_name_en='The market determines what is coming',
               choice_name_nl='De markt bepaalt wat er komt', context_id=context.id),
        Choice(choice_order=4, choice_name_en='Large-scale energy generation will occur in a small number of places',
               choice_name_nl='Op een klein aantal plekken komt grootschalige energieopwekking', context_id=context.id),
        Choice(choice_order=5, choice_name_en='Betting on storage',
               choice_name_nl='Inzetten op opslag', context_id=context.id),
        Choice(choice_order=6, choice_name_en='Becoming an energy supplier in the Netherlands',
               choice_name_nl='Energieleverancier van Nederland worden', context_id=context.id),
    ]
    db.session.add_all(choices)
    db.session.commit()

    # write motivations to db
    parse_csv(motivations_filename)
    db.session.commit()


def parse_csv(filename):
    import csv
    from app.models import Motivation

    with open(filename, newline='') as motivations_file:
        reader = csv.DictReader(motivations_file, delimiter=',')
        for i, row in enumerate(reader):

            db.session.add(Motivation(
                    pve_idx=i,
                    motivation_en=row['message'],
                    motivation_nl="none",
                    choice_id=2  # arbitrary number < number of choices
                ))


# write ukraine data to db
if init_data:
    app.app_context().push()
    init_ukraine_context("")  # insert path to local file
